from __future__ import division, print_function

import opticstools as ot
import numpy as np
import matplotlib.pyplot as plt
import pdb
import scipy.ndimage as nd
import scipy.linalg as la
import matplotlib.cm as cm
import miscutils as mu
import time

TENTH_AIRY_RING = 10.25

plt.ion()

def strehl(psf, psf_dl):
	""" Calculate the Strehl ratio of an aberrated input PSF given the diffraction-limited PSF. """
	return np.amax(psf) / np.amax(psf_dl)
	
def centre_crop(psf, psf_sz_cropped):
    """Crop an image about the center"""
    return psf[psf.shape[0]/2-psf_sz_cropped//2:psf.shape[0]/2-psf_sz_cropped//2+psf_sz_cropped + np.mod(psf_sz_cropped,2),
               psf.shape[1]/2-psf_sz_cropped//2:psf.shape[1]/2-psf_sz_cropped//2+psf_sz_cropped + np.mod(psf_sz_cropped,2)] 
	
class SCFeedBackAO():
    """A single-congugate adaptive optics system. The initialization order should be:
    
    1) Wavefront() for every wavelength.
    2) A wavefront sensor.
    3) A deformable mirror.
    4) This AO system.
    5) An Atmosphere().
    
    By convention, if not set, all wavefronts not used for sensing are
    used to create separate science images.
    
    Parameters
    ----------
    dm: DeformableMirror instance, which contains all wavefronts.
    wfs: WFS or child class instance
    atm: A single atmosphere for all wavefronts.
    conjugate_location: float
        location of wfs and DM conjugate
    image_ixs: list
        A list of indexes for which images should be calculated, i.e. the 
        science wavefront indices in the dm wavefronts list.
    dm_poke_scale: float
        A normalisation for poking the deformable mirror, used in response matrix
        sensing and the loop. Should be well within the linear regime."""

    def __init__(self,
                dm,
                wfs,
                atm,
                conjugate_location=0.0,
                image_ixs=None,
                dm_poke_scale=1e-7):
        
        # A list of indexes for which images should be calculated, i.e. the science wavefront indices from the dm instance.
        # If no image indexes are specified, then it is assumed that all wavefronts in the DM
        # that are not sensed by the WFS are used for imaging.
        if not image_ixs:
            image_ixs = []            
            for j in range(len(dm.wavefronts)):
                if dm.wavefronts[j] not in wfs.wavefronts:
                    image_ixs.append(j)
        elif type(image_ixs) == int:
            image_ixs = [image_ixs]
        elif type(image_ixs) != list:
            print("OOPS: invalid type for image_ixs - must be of type list or int!")
            raise UserWarning

        # Conjugate location
        if conjugate_location != 0:
            print("OOPS: Not implemented yet - only ground layer conugation so far")
            raise UserWarning
        # location of WFS and DM conjugate
        self.conjugate_location=conjugate_location
        
        self.image_ixs = image_ixs
        self.wavefronts = dm.wavefronts
        self.dm = dm
        self.wfs = wfs        
        self.atm = atm
        self.dm_poke_scale = dm_poke_scale

        #Add an atmosphere to all wavefronts.
        for wf in self.wavefronts:
            wf.add_atmosphere(atm)
        
    #############################################################################################################
    def psf_dl(self, plate_scale_as_px,
        ix = None,
        band = None,
        crop = True,    # Whether or not to crop the PSF. If set to False, the below two arguments are irrelevant.
        psf_sz_cropped = None,          # Size of the PSF. By default cropped at the 10th Airy ring
        psf_sigma_limit_N_os = TENTH_AIRY_RING,     # Corresponds to 10 Airy rings
        plotit = False
        ):
        """ 
            Compute and return the diffraction-limited PSF at the specified wavelength and plate scale. 

            By default, the returned PSF is cropped at the 10th Airy ring. 
        """

        print("Generating the diffraction-limited PSF at wavelength {:.2f} nm...".format(self.wavefronts[ix].wave * 1e9))

        # Generating the PSF
        psf = self.wavefronts[ix].psf_dl(plate_scale_as_px = plate_scale_as_px, plotit = plotit, return_efield = False)
        psf_sz = psf.shape[0]

        # If crop == False, we simply return the PSF at the native size generated by the FFT method that is used to compute the PSF.
        if crop:           
            # How many 'sigmas' we want the returned PSF grid to encompass. By default, the PSF is cropped at the 10th Airy ring. 
            if psf_sz_cropped == None:
                N_OS = self.wavefronts[ix].wave / self.wavefronts[ix].D / 2 / np.deg2rad(plate_scale_as_px / 3600)  # Nyquist sampling (HWHM in pixels)
                psf_sz_cropped = np.ceil(min(psf_sz, 4 * N_OS * psf_sigma_limit_N_os))
                # This needs to be a tuple for the below lines to work (in case of different dimensions in x and y)
                psf_sz_cropped = (psf_sz_cropped, psf_sz_cropped)
           
            # If a scalar value is supplied, we crop to a square. 
            elif matplotlib.cbook.is_scalar(psf_sz_cropped):
                psf_sz_cropped = (psf_sz_cropped, psf_sz_cropped)
           
            psf = centre_crop(psf, psf_sz_cropped) 

        if plotit:
            mu.astroimshow(im = psf, plate_scale_as_px = plate_scale_as_px, title = r'Diffraction-limited PSF at $\lambda = %.1f$ $\mu$m' % (self.wavefronts[ix].wave * 1e9))

        return psf

    #############################################################################################################
    def find_response_matrix(self,mode='onebyone',amplitude=1e-7):
        """Poke the actuators and record the WFS output"""
        self.response_matrix = np.empty( (self.dm.nactuators,self.wfs.nsense) )
        print("Computing AO system response matrix.")
        if mode=='onebyone':
            # We poke each actuator twice (once +ve, once -ve) and take the abs. mean WFS response
            print(".",end="")
            for i in range(self.dm.nactuators):
                
                # Flatten the WFS wavefronts
                for wf in self.wfs.wavefronts:
                    # wf.field = wf.pupil
                    wf.flatten_field()
                
                # Poking an actuator in the +ve direction.
                act = np.zeros(self.dm.nactuators)                
                act[i] = self.dm_poke_scale
                self.dm.apply(act)          # Apply the poke to the wavefront, perturb its phase
                wfs_plus = self.wfs.sense() # Get the corresonding WFS measurement
                
                # Flatten the WFS wavefronts
                for wf in self.wfs.wavefronts:
                    # wf.field = wf.pupil
                    wf.flatten_field()

                # Poking an actuator in the -ve direction.
                act = np.zeros(self.dm.nactuators)
                act[i] = -self.dm_poke_scale
                self.dm.apply(act)
                wfs_minus = self.wfs.sense() 

                #Check that poking in both directions is equivalent!
                if (np.sum(wfs_plus*wfs_minus)/np.sum(wfs_plus*wfs_plus) > -0.9):
                    print("WARNING: Poking the DM is assymetric!!!")
                    # pdb.set_trace()

                # Taking the mean response value.
                self.response_matrix[i] = 0.5*(wfs_plus - wfs_minus).flatten()
        else:
            print("ERROR: invalid response matrix mode")
            raise UserWarning
        self.response_matrix = self.response_matrix.T
        
    #############################################################################################################
    def compute_reconstructor(self,mode='eigenclamp', threshold=0.2):
        """Compute the reconstructor
        
        Parameters
        ----------
        mode: string
            Mode for this calculation. 'eigenclamp' computes a Moore-Penrose
            pseudo-inverse with SVD eigenvalue clamping.
        threshold: float
            Threshold for eigenvalue clamping.
        """
        #!!! for speed, with larger problems, full_matrices should *not* be 
        # true. But then we have to think more carefully about the math!
        u,s,v = la.svd(self.response_matrix,full_matrices=True)
        if mode=='eigenclamp':
            bad_modes = s < np.mean(s)*threshold
            s[bad_modes] = np.inf
            self.reconstructor = np.dot(np.dot(u,la.diagsvd(1.0/s,len(u),len(v))),v).T
        else:
            print("ERROR: reconstructor mode")
            raise UserWarning
        
    #############################################################################################################   
    def correct_twice(self,plotit=False):
        """Find the pupil field, then correct it twice.
        
        TEST method, but ERROR checking still needed!
        
        Returns
        -------
        measurements: list
            A list of the sensor output for before correction, after 1 and after 2
            applications of the reconstructor
        ims: list
            A list of wavefront sensor images (before correction, after 1 and 2
            reconstructor applications)
        """
        # Reset the distorted wavefront to that caused by the atmosphere and nothing else
        for wf in self.wfs.wavefronts:
            wf.atm_field()    # Reset the pupil field
            field0 = wf.field

        # Sense the wavefront.
        # wfs.sense() uses the field attribute of the wf to make a measurement.
        measurements0 = self.wfs.sense() # WF measurements
        im0 = self.wfs.im.copy()    # WFS image
        # Calculate the DM coefficients corresponding to the measurements.
        coefficients = -np.dot(self.reconstructor,measurements0.flatten())*self.dm_poke_scale
        
        # Reset the wavefront (since it gets modified by wfs.sense())
        for wf in self.wfs.wavefronts:
            wf.atm_field()
            field1 = wf.field
        
        # Apply a correction. This modifies (but does not reset) the 
        # field attribute of wf:
        #   wf.field = wf.field*np.exp(2j*np.pi*phasescreen/wf.wave)
        self.dm.apply(coefficients)
        
        # Sense after the first correction
        measurements1 = self.wfs.sense()
        im1 = self.wfs.im.copy()
        coefficients += -np.dot(self.reconstructor,measurements1.flatten())*self.dm_poke_scale
        
        for wf in self.wfs.wavefronts:
            wf.atm_field()
            field2 = wf.field
        
        # Apply a second correction
        # self.dm.apply(coefficients)
        
        # Sense after the second correction.
        measurements2 = self.wfs.sense()
        im2 = self.wfs.im.copy()

        if plotit==True:
            plt.figure()
            plt.suptitle('WFS detector images')
            plt.subplot(131)
            plt.imshow(im0)
            plt.title('Before sensing')
            plt.subplot(132)
            plt.imshow(im1)
            plt.title('After first correction')
            plt.subplot(133)
            plt.imshow(im2)
            plt.title('After second correction')

        return [measurements0,measurements1,measurements2],[im0,im1,im2] 
        
    #############################################################################################################
    def run_loop(self, dt, 
                mode = 'integrator',
                niter = 100,
                gains = {
                    'K_p' : 0.0, 
                    'K_i' : 1.0, 
                    'K_d' : 0.0, 
                    'K_leak' : 0.9
                    },
                psf_ix = 0,                     # Index in the DM's list of wavefronts of the PSF you want to be returned
                psf_sz_cropped = None,          # Size of the PSF. By default cropped at the 10th Airy ring
                psf_sigma_limit_N_os = TENTH_AIRY_RING,     # Corresponds to 10 Airy rings
                plate_scale_as_px = None,       # Plate scale of the output images/PSFs
                plot_sz_px = (80,80),     # For now this is only used in plotting
                nframesbetweenplots = 10,
                plotit = False
                ):
        """Run an AO servo loop.
        
        It is assumed that all wavefronts share the same atmosphere!
        
        Parameters
        ----------
        dt: float
            Time between samples
        mode: string
            Servo loop mode. 
            * 'integrator' is classic integral control with parameters K_i and K_leak.
            * 'PID' is simple proportional-integral-derivative control. 
            * 'open_loop' involves no wavefront correction (only sensing).
            For PD, PI, etc. control, simply set the unneeded gain terms to zero.
        niter: int
            Number of iterations of the loop.
        plotit: boolean
            Do we plot potentially useful outputs? NB, this is all pretty slow with
            matplotlib.

        K_i: float
            Integral gain. 
        K_p: float
            Proportional gain.
        K_d: float
            Derivative gain.
        K_leak: float
            Leak coefficient for integral control. (allows transient errors in WFS 
            measurements to leak out of the control input over time)
            Is automatically zeroed if K_i is zeroed. 
        psf_ix: int
            Index in self.image_ixs corresponding to the wavelength of the PSF to be returned.
        plate_scale_as_px: float
            Plate scale for the output science images. For now, only put in the plate scale instead of the Nyquist sampling because we image at a range of wavelengths with the same detector.
        """

        # Zero the leak coefficient if no integral control is used. 
        # DM coefficients (control commands)
        coefficients_current = np.zeros(self.dm.nactuators) # timestep k
        
        # WFS measurements
        y_integral = np.zeros(self.wfs.nsense) 
        y_current = np.zeros(self.wfs.nsense)    # timestep k
        y_old = np.zeros(self.wfs.nsense)        # timestep k - 1
        
        # Figuring out the size of the PSF image to return.
        psf_sz = self.wavefronts[psf_ix].image(plate_scale_as_px=plate_scale_as_px).shape[0]
        if plate_scale_as_px and not psf_sz_cropped:
            # Oversampling factor of the PSF. Note that N_OS is equivalent to the 'sigma' of the pupil image in the DL.
            plate_scale_rad_px = np.deg2rad(plate_scale_as_px / 3600)
            N_OS = self.wavefronts[psf_ix].wave / self.wavefronts[psf_ix].D / 2 / plate_scale_rad_px
            psf_sz_cropped = np.ceil(min(psf_sz, 4 * N_OS * psf_sigma_limit_N_os))
        else:
            if psf_sz_cropped:
                psf_sz_cropped = np.ceil(min(psf_sz, psf_sz_cropped))        
            else:
                psf_sz_cropped = psf_sz

        # Arrays to hold images
        psfs_cropped = np.zeros((niter, psf_sz_cropped, psf_sz_cropped))# at the psf_ix wavelength

        """ AO Control loop """
        print("Starting the AO control loop with control logic mode '%s'..." % mode)
        for k in range(niter):  
            #------------------ AO CONTROL ------------------#
            print("Iteration %d..." % (k+1))
            # Evolve the atmosphere & update the wavefront fields to reflect the new atmosphere.                
            self.atm.evolve(dt * k)
            for wf in self.wavefronts:   
                wf.atm_field() 
            
            if mode == 'open loop':
                # We still measure the wavefront for plotting.
                self.wfs.sense()
            else:
                self.dm.apply(coefficients_current) # Apply the wavefront correction. Note that this does NOT reset the wavefront but simply applies the phase change to the field variable.       
                measurements = self.wfs.sense() # Sense the corrected wavefront. Does not modify the field variable.
                y_old = y_current                       # y at timestep k - 1
                y_current = measurements.flatten()      # y at timestep k
                y_integral += y_current * dt            # Integrate over time.
                coefficients_next = np.zeros(self.dm.nactuators)    # timestep k + 1
                # Apply control logic.                
                if mode == 'PID':
                    coefficients_next +=  - gains['K_i'] * np.dot(self.reconstructor,y_integral) * self.dm_poke_scale
                    coefficients_next += gains['K_leak'] * coefficients_current - gains['K_p'] * np.dot(self.reconstructor,y_current) * self.dm_poke_scale
                    coefficients_next += - gains['K_d'] * (1/dt) * (np.dot(self.reconstructor,y_current) - np.dot(self.reconstructor,y_old)) * self.dm_poke_scale
                elif mode == 'integrator':  
                    coefficients_next = gains['K_leak'] * coefficients_current - gains['K_i'] * np.dot(self.reconstructor,y_current) * self.dm_poke_scale
                else:
                    print("ERROR: invalid control logic specified!")
                    raise UserWarning            
                coefficients_current = coefficients_next;   # Update the DM coefficients.

            # Create the PSF            
            psf = self.wavefronts[psf_ix].image(plate_scale_as_px = plate_scale_as_px) 
            psf /= sum(psf.flatten())             
            # Saving the PSF.            
            psfs_cropped[k] = centre_crop(psf, psf_sz_cropped) 
                   
            # Plotting
            if plotit & ((k % nframesbetweenplots) == 0):
                self._plot_ao_loop_iteration(k, plot_sz_px)
                
        return psfs_cropped

    #############################################################################################################
    def _plot_ao_loop_iteration(self, k, plot_sz_px):
        """ Plot the WFS detector image, phase and psf in an iteration of the AO loop. """
        if k == 0:
            axes = []
            plots = []
            plt.rc('text', usetex=True)

            # WFS plot                    
            fig_wfs = mu.newfigure(2,2)
            ax_wfs = fig_wfs.add_subplot(111)  # WFS detector image
            ax_wfs.title.set_text(r'WFS detector')
            ax_wfs.axis( [0,self.wavefronts[0].sz,0,self.wavefronts[0].sz] )
            plot_wfs = ax_wfs.imshow(self.wfs.im,interpolation='nearest',cmap=cm.gray)

            fig = mu.newfigure(width=2,height=1)
        
            # Phase
            axes.append(fig.add_subplot(N_science_imgs,2,2*j+1)) 
            axes[-1].title.set_text(r'Corrected phase ($\lambda$ = %d nm)' % (self.wavefronts[self.image_ixs[j]].wave*1e9))
            plots.append(axes[-1].imshow(np.angle(self.wavefronts[self.image_ixs[j]].field)*self.wavefronts[self.image_ixs[j]].pupil,interpolation='nearest', cmap=cm.gist_rainbow, vmin=-np.pi, vmax=np.pi))
            mu.colourbar(plots[-1])

            # Science image
            axes.append(fig.add_subplot(N_science_imgs,2,2*j+2))  
            axes[-1].title.set_text(r'Science Image ($\lambda$ = %d nm)' % (self.wavefronts[self.image_ixs[j]].wave*1e9))
            pad_x = max((imsize - plot_sz_px[0]) // 2, 0)
            pad_y = max((imsize - plot_sz_px[1]) // 2, 0)
            im_cropped = psf_science[j,pad_x:pad_x + min(imsize, plot_sz_px[0]),pad_y:pad_y + min(imsize, plot_sz_px[1])]
            plots.append(axes[-1].imshow(im_cropped,interpolation='nearest', cmap=cm.gist_heat, norm=LogNorm()))
            mu.colourbar(plots[-1])
        else:
            # Update the plots
            plot_wfs.set_data(self.wfs.im)  
            fig.suptitle(r'AO-corrected phase and science images, $k = %d, K_i = %.2f, K_{leak} = %.2f$' % (k, K_i, K_leak))
            for j in range(N_science_imgs):
                plots[2*j].set_data(np.angle(self.wavefronts[self.image_ixs[j]].field)*self.wavefronts[self.image_ixs[j]].pupil)
                pad_x = max((imsize - plot_sz_px[0]) // 2, 0)
                pad_y = max((imsize - plot_sz_px[1]) // 2, 0)
                im_cropped = im_science[j,pad_x:pad_x + min(imsize, plot_sz_px[0]),pad_y:pad_y + min(imsize, plot_sz_px[1])]
                plots[2*j+1].set_data(im_cropped)

        plt.draw()
        plt.pause(0.00001)  # Need this to plot on some machines.


        