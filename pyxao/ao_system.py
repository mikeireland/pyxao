from __future__ import division, print_function
from aosim.pyxao import *
from linguinesim.apdsim import convolvePSF
import opticstools as ot
import numpy as np
import matplotlib.pyplot as plt
import pdb
import scipy.ndimage as nd
import scipy.linalg as la
import matplotlib.cm as cm
import time
plt.ion()

class SCFeedBackAO():
    """A single-congugate adaptive optics system. The initialization order should be:
    
    1) Wavefront() for every wavelength.
    2) A wavefront sensor.
    3) A deformable mirror.
    4) This AO system.
    5) An Atmosphere() for every wavefront.
    
    By convention, if not set, all wavefronts not used for sensing are
    used to create separate science images.
    
    Parameters
    ----------
    dm: DeformableMirror instance
    wfs: WFS or child class instance
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
        self.dm_poke_scale = dm_poke_scale
        
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
            
    def correct_twice(self,plotIt=False):
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

        if plotIt==True:
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
        
    def run_loop(self, dt, 
                mode = 'open_loop',
                niter = 100,
                gains = {
                    'K_p' : 0.5, 
                    'K_i' : 1.0, 
                    'K_d' : 0.2, 
                    'K_leak' : 0.9
                    },
                psf_ix = 0,                     # Index in the list of wavefronts of the PSF you want to be returned
                psf_sz_cropped = None,          # Size of the PSF. By default cropped at the 10th Airy ring
                plate_scale_as_px = None,       # Plate scale of the output images/PSFs
                im_input = None,                 # Input image that we convolve with the PSF at the wavelength corresponding to psf_ix. For now it is assumed to have the same plate scale as plate_scale_as_px.
                detector_size_px = (40,40),    # For now this is only used in plotting
                nframesbetweenplots = 10,
                plotIt = False
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
        plotIt: boolean
            Do we plot potentially useful outputs? NB, this is all pretty slow with
            matplotlib.
        gains: dictionary
            Gains for the control loop.
            K_i: float
                Integral gain. 
            K_p: float
                Proportional gain.
            K_d: float
                Derivative gain.
            K_leak: float
                Leak coefficient for integral control. (allows transient errors in WFS 
                measurements to leak out of the control input over time)
        psf_ix: int
            Index in self.image_ixs corresponding to the wavelength of the PSF to be returned.
        plate_scale_as_px: float
            Plate scale for the output science images. For now, only put in the plate scale instead of the Nyquist sampling because we image at a range of wavelengths with the same detector.
        """
        # DM coefficients (control commands)
        coefficients_current = np.zeros(self.dm.nactuators) # timestep k
        
        # WFS measurements
        y_current = np.zeros(self.wfs.nsense)    # timestep k
        y_old = np.zeros(self.wfs.nsense)        # timestep k - 1

        # Size of the largest image that will be generated.        
        N_science_imgs = len(self.image_ixs)
        imsize = 0
        for ix in self.image_ixs:
          imsize = max(imsize, self.dm.wavefronts[ix].image(plate_scale_as_px=plate_scale_as_px).shape[0])  
        
        # Size of the uncropped PSF image.
        psf_sz = self.dm.wavefronts[psf_ix].image(plate_scale_as_px=plate_scale_as_px).shape[0]
        # Oversampling factor of the PSF. Note that N_OS is equivalent to the 'sigma' of the pupil image in the DL.
        plate_scale_rad_px = np.deg2rad(plate_scale_as_px / 3600)
        N_OS = self.dm.wavefronts[psf_ix].wave / self.dm.wavefronts[psf_ix].D / 2 / plate_scale_rad_px
        # How many 'sigmas' we want the returned PSF grid to encompass.
        if not psf_sz_cropped:
            psf_sigma_limit_px = 10.25  # Corresponds to 10 Airy rings
            psf_sz_cropped = np.ceil(min(psf_sz, 4 * N_OS * psf_sigma_limit_px))
        else:
            psf_sz_cropped = np.ceil(min(psf_sz, psf_sz_cropped))        
        # Arrays to hold images
        psfs_cropped = np.zeros((niter, psf_sz_cropped, psf_sz_cropped))# at the psf_ix wavelength
        psf_mean = np.zeros((psf_sz_cropped, psf_sz_cropped))           # at the psf_ix wavelength
        psf_mean_all = np.zeros((imsize, imsize))   # sum over all wavelengths
        # If an input image is specified.
        if im_input is not None:
            # If the image size is actually smaller than the detector, then we make the detector this size too.
            detector_size_px = (min(im_input.shape[0], detector_size_px[0]), min(im_input.shape[1], detector_size_px[1]))
            im_mean = np.zeros(detector_size_px)    # at the psf_ix wavelength
            ims_science = np.zeros((niter, detector_size_px[0], detector_size_px[1]))    # at the psf_ix wavelength
            im_input_cropped = mu.centreCrop(im_input, detector_size_px)    # Cropped to detector size

        """ AO Control loop """
        # At this point, the field variables all include the atmosphere.
        # Order in loop is
        # update atmosphere --> apply DM correction --> sense wavefront --> update coefficients
        print("Starting the AO control loop with control logic mode '%s'..." % mode)
        for k in range(niter):  
            #------------------ AO CONTROL ------------------#
            print("Iteration %d..." % (k+1))
            for wf in self.dm.wavefronts:   # Evolve the atmosphere & update the wavefront fields to reflect the new atmosphere.                
                wf.atm.evolve(dt * k)   # This doesn't affect the field variable.
                wf.atm_field()          # Now, the wavefront fields include only the atmosphere.           
            
            if mode == 'open_loop':
                # We still measure the wavefront for plotting.
                self.wfs.sense()
            else:
                self.dm.apply(coefficients_current) # Apply the wavefront correction. Note that this does NOT reset the wavefront but simply applies the phase change to the field variable.       
                measurements = self.wfs.sense() # Sense the corrected wavefront. Does not modify the field variable.
                y_old = y_current                       # y at timestep k - 1
                y_current = measurements.flatten()      # y at timestep k
                coefficients_next = np.zeros(self.dm.nactuators)    # timestep k + 1
                # Apply control logic.                
                if mode == 'PID':
                    coefficients_next += gains['K_leak'] * coefficients_current - gains['K_i'] * np.dot(self.reconstructor,y_current) * self.dm_poke_scale
                    coefficients_next += - gains['K_p'] * np.dot(self.reconstructor,y_current) * self.dm_poke_scale
                    coefficients_next += - gains['K_d'] * (1/dt) * (np.dot(self.reconstructor,y_current) - np.dot(self.reconstructor,y_old)) * self.dm_poke_scale
                elif mode == 'integrator':  
                    coefficients_next = gains['K_leak'] * coefficients_current - gains['K_i'] * np.dot(self.reconstructor,y_current) * self.dm_poke_scale
                else:
                    print("ERROR: invalid control logic specified!")
                    raise UserWarning            
                coefficients_current = coefficients_next;   # Update the DM coefficients.
                

            #-------------------- SAVING PSFS AND SCIENCE IMAGE ----------------------#
            # Create the PSFs. 
            psf_science_all = np.zeros((imsize, imsize))
            psfs_science = np.zeros((N_science_imgs, imsize, imsize))
            for i in range(N_science_imgs):
                ix = self.image_ixs[i]
                psf = self.dm.wavefronts[ix].image(plate_scale_as_px = plate_scale_as_px) 
                psf /= sum(psf.flatten())               
                # Saving the PSF.
                if ix == psf_ix:
                    psfs_cropped[k] = mu.centreCrop(psf, psf_sz_cropped) 
                pad_x = (imsize - psf.shape[0])//2
                pad_y = (imsize - psf.shape[1])//2       
                psfs_science[i,pad_x:pad_x + psf.shape[0], pad_y:pad_y + psf.shape[1]] = psf  # Science images separated into different wavelengths
                psf_science_all += psfs_science[i] / N_science_imgs   # Stacked science image.       
            
            # Update the mean PSF.
            if (k != 0):
                psf_mean += psfs_cropped[k]
                psf_mean_all += psf_science_all  

            # Create the science image, if required.
            # The science image is im_input (cropped to the detector size) convolved with the PSF corresponding to psf_ix.
            if im_input is not None:
                # pdb.set_trace()
                im_science = apdsim.convolvePSF(image = im_input_cropped, psf = psfs_cropped[k])
                ims_science[k] = im_science
                # Update the mean image.
                if (k != 0):
                    im_mean += im_science           
                 
            #------------------ PLOTTING ------------------#
            if plotIt & ((k % nframesbetweenplots) == 0):                  
                if k == 0:
                    axes = []
                    plots = []
                    if im_input is not None:
                        fig_width = 4
                    else:
                        fig_width = 3
                    plt.rc('text', usetex=True)
                    fig = mu.newfigure(width=fig_width,height=N_science_imgs)                    
                    for j in range(N_science_imgs):
                        # WFS 
                        axes.append(fig.add_subplot(N_science_imgs,fig_width,2*j+1)) 
                        axes[-1].title.set_text(r'Wavefront sensor detector image')
                        plots.append(axes[-1].imshow(self.wfs.im,interpolation='nearest',cmap=cm.gray))
                        mu.colourbar(plots[-1])
                        # Phase
                        # Get the current phase for plotting.
                        phase_current = np.angle(self.dm.wavefronts[self.image_ixs[j]].field)*self.dm.wavefronts[self.image_ixs[j]].pupil
                        axes.append(fig.add_subplot(N_science_imgs,fig_width,2*j+2)) 
                        axes[-1].title.set_text(r'Corrected phase ($\lambda$ = %d nm)' % (self.dm.wavefronts[self.image_ixs[j]].wave*1e9))
                        plots.append(axes[-1].imshow(phase_current,interpolation='nearest', cmap=cm.gist_rainbow, vmin=-np.pi, vmax=np.pi))
                        mu.colourbar(plots[-1])
                        # PSF
                        axes.append(fig.add_subplot(N_science_imgs,fig_width,2*j+3))  
                        axes[-1].title.set_text(r'Point spread function ($\lambda$ = %d nm)' % (self.dm.wavefronts[self.image_ixs[j]].wave*1e9))
                        psf_cropped = mu.centreCrop(psfs_science[j], detector_size_px)
                        plots.append(axes[-1].imshow(psf_cropped,interpolation='nearest', cmap=cm.gist_heat))
                        mu.colourbar(plots[-1])
                        # Science image
                        if im_input is not None and psf_ix == self.image_ixs[j]:
                            axes.append(fig.add_subplot(N_science_imgs,fig_width,2*j+4))  
                            axes[-1].title.set_text(r'Science Image ($\lambda$ = %d nm)' % (self.dm.wavefronts[self.image_ixs[j]].wave*1e9))    
                            plots.append(axes[-1].imshow(im_science, interpolation='nearest', cmap=cm.gist_heat))
                            mu.colourbar(plots[-1])
                else:
                    if mode == 'open_loop':
                        fig.suptitle(r'Open loop phase and science images, $k = %d$' % k)
                    else:
                        fig.suptitle(r'AO-corrected phase and science images, $k = %d$' % k)
                    for j in range(N_science_imgs):
                        plots[2*j].set_data(self.wfs.im)        # Update WFS                        
                        phase_current = np.angle(self.dm.wavefronts[self.image_ixs[j]].field)*self.dm.wavefronts[self.image_ixs[j]].pupil
                        plots[2*j+1].set_data(phase_current)    # Update phase
                        psf_cropped = mu.centreCrop(psfs_science[j], detector_size_px)   # Update PSF. Crop it to the detector size first 
                        plots[2*j+2].set_data(psf_cropped)                        
                        if im_input is not None:                # Update image
                            plots[2*j+3].set_data(im_science)

                plt.draw()
                plt.pause(0.00001)  # Need this to plot on some machines.
            #----------------------------------------------#

        # Compute the mean images.
        psf_mean /= niter
        psf_mean_all /= niter      

        if plotIt:
            mu.newfigure(1,3)
            plt.suptitle('AO PSFs, control mode %s' % mode)
            mu.astroimshow(mu.centreCrop(self.dm.wavefronts[psf_ix].psf_dl(plate_scale_as_px = plate_scale_as_px), psf_sz_cropped), 'Diffraction-limited PSF', plate_scale_as_px, 131)
            mu.astroimshow(psf_mean, r'Mean PSF ($\lambda = %d$ nm)' % (self.dm.wavefronts[psf_ix].wave*1e9), plate_scale_as_px, 132)
            mu.astroimshow(mu.centreCrop(psf_mean_all, psf_sz_cropped), 'Mean PSF (all wavelengths)', plate_scale_as_px, 133)
            plt.show()              

        if im_input is not None:
            im_mean /= niter
            return psfs_cropped, ims_science, psf_mean, im_mean, psf_mean_all # psf_perfect
        else:
            return psfs_cropped, psf_mean, psf_mean_all # psf_perfect