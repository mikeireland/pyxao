from __future__ import division, print_function
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
        science wavefront indices from the dm instance.
    dm_poke_scale: float
        A normalisation for poking the deformable mirror, used in response matrix
        sensing and the loop. Should be well within the linear regime."""

    def __init__(self,dm,wfs,conjugate_location=0.0,image_ixs=None,
        # A normalisation for poking the deformable mirror, used in response matrix sensing and the loop. 
        # Should be well within the linear regime.
        dm_poke_scale=1e-7):
        
        # A list of indexes for which images should be calculated, i.e. the science wavefront indices from the dm instance.
        if not image_ixs:
            image_ixs = range(len(dm.wavefronts),len(wfs.wavefronts))
        # AZ: make sure image_ixs is a list, even if only with one element
        elif type(image_ixs) == int:
            image_ixs = [image_ixs]
        else:
            print("OOPS: invalid type for image_ixs - must be of type list or int!")
            raise UserWarning

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
        if mode=='onebyone':
            # We poke each actuator twice (once +ve, once -ve) and take the abs. mean WFS response
            for i in range(self.dm.nactuators):
                
                #Flatten the WFS wavefronts
                for wf in self.wfs.wavefronts:
                    wf.field=wf.pupil
                
                # Poking an actuator in the +ve direction.
                act = np.zeros(self.dm.nactuators)                
                act[i] = self.dm_poke_scale
                self.dm.apply(act)          # Apply the poke to the wavefront, perturb its phase
                wfs_plus = self.wfs.sense() # Get the corresonding WFS measurement
                
                #Flatten the WFS wavefronts
                for wf in self.wfs.wavefronts:
                    wf.field=wf.pupil

                # Poking an actuator in the -ve direction.
                act = np.zeros(self.dm.nactuators)
                act[i] = -self.dm_poke_scale
                self.dm.apply(act)
                wfs_minus = self.wfs.sense() 

                #Check that poking in both directions is equivalent!
                if (np.sum(wfs_plus*wfs_minus)/np.sum(wfs_plus*wfs_plus) > -0.9):
                    print("WARNING: Poking the DM is assymetric!!!")
                    pdb.set_trace()

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
            wf.pupil_field()    # Reset the pupil field
            field0 = wf.field

        # Sense the wavefront.
        # wfs.sense() uses the field attribute of the wf to make a measurement.
        measurements0 = self.wfs.sense() # WF measurements
        im0 = self.wfs.im.copy()    # WFS image
        # Calculate the DM coefficients corresponding to the measurements.
        coefficients = -np.dot(self.reconstructor,measurements0.flatten())*self.dm_poke_scale
        
        # Reset the wavefront (since it gets modified by wfs.sense())
        for wf in self.wfs.wavefronts:
            wf.pupil_field()
            field1 = wf.field
        
        # Apply a correction. This modifies (but does not reset) the 
        # field attribute of wf:
        #   wf.field = wf.field*np.exp(2j*np.pi*phasescreen/wf.wave)
        # We have to reset the wavefront using
        # pupil_field() before we call this because field is modified by
        # wfs.sense() (gets masked with the lenslet pupils)
        self.dm.apply(coefficients)
        
        # Sense after the first correction
        measurements1 = self.wfs.sense()
        im1 = self.wfs.im.copy()
        coefficients += -np.dot(self.reconstructor,measurements1.flatten())*self.dm_poke_scale
        
        for wf in self.wfs.wavefronts:
            wf.pupil_field()
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

            """
            # Sanity check to see how the field changes between calls of 
            # pupil_field(). (hint: it doesn't)

            plt.subplot(131)
            plt.imshow(field0.real)
            plt.title('field0')
            plt.subplot(132)
            plt.imshow(field1.real)
            plt.title('field1')
            plt.subplot(133)
            plt.imshow((field1-field0).real)
            plt.title('field1-field0')
            plt.colorbar()
            """

            """
            plt.figure()
            plt.suptitle('WFS detector image changes')
            plt.subplot(121)
            plt.imshow(im1-im0)
            plt.title('im1 - im0')
            plt.colorbar()
            plt.subplot(122)
            plt.imshow(im2-im1)
            plt.title('im2 - im1')
            plt.colorbar()
            """

        return [measurements0,measurements1,measurements2],[im0,im1,im2] 
        
    def run_loop(self,dt=0.002,nphot=1e4,mode='integrator',niter=1000,nframesbetweenplots=10,plotit=False,\
        gain=1.0,dodgy_damping=0.9):
        """Run an AO servo loop.
        
        It is assumed that all wavefronts share the same atmosphere!
        
        Parameters
        ----------
        dt: float
            Time between samples
        nphot: float
            Number of photons per frame (noise)
        mode: string
            Servo loop mode. 'integrator' is a simple integrator.
        niter: int
            Number of iterations of the loop.
        plotit: boolean
            Do we plot potentially useful outputs? NB, this is all pretty slow with
            matplotlib.
        gain: float
            Servo loop gain. 
        dodgy_damping: float
            Due to Mike's lack of understanding of servo theory, damping the DM
            position to zero in the absence of WFS seems to help.
        """
        coefficients_current = np.zeros(self.dm.nactuators)
        sz = self.dm.wavefronts[-1].sz
        im_mn = np.zeros( (2*sz,2*sz) )

        for i in range(niter):
            # Evolve the atmosphere.
            self.dm.wavefronts[0].atm.evolve(dt*i)
            
            # Reset the pupil fields.
            for wf in self.dm.wavefronts:
                wf.pupil_field()
            
            # Apply the wavefront correction.
            self.dm.apply(coefficients_current)
            
            # Save a copy of the corrected pupil phase.
            corrected_field = self.dm.wavefronts[0].field.copy()
            
            #Sense the corrected wavefront.
            sensors = self.wfs.sense(nphot=nphot)
            
            # Apply control logic.
            # pdb.set_trace()
            if mode=='integrator':
                coefficients_current = dodgy_damping*coefficients_current - gain*np.dot(self.reconstructor,sensors.flatten())*self.dm_poke_scale
            else:
                print('ERROR: invalid control loop mode! Use integrator for now...')
                raise UserWarning

            # Create the image. By FFT for now...
            im_science = np.zeros( (sz*2,sz*2) )

            # AZ
            # for ix in range(self.image_ixs):
            for ix in self.image_ixs:
                # Huygens propagation to generate the science image (i.e. FFT)
                im_science += self.dm.wavefronts[ix].image()
            im_mn += im_science 
            
            # Plot stuff if we want.
            if plotit & ((i % nframesbetweenplots)==0):
                plt.clf()
                # Plot the WFS detector image
                plt.subplot(131)
                plt.imshow(self.wfs.im,interpolation='nearest',cmap=cm.gray)
                # plt.plot(self.wfs.px[:,0], self.wfs.px[:,1],'x')
                plt.axis( [0,self.dm.wavefronts[0].sz,0,self.dm.wavefronts[0].sz] )
                plt.title('WFS')
                # Plot the corrected phase 
                plt.subplot(132)
                plt.imshow(np.angle(corrected_field)*self.dm.wavefronts[0].pupil,interpolation='nearest')
                plt.title('Corrected Phase')
                # Plot the science image
                plt.subplot(133)
                plt.imshow(im_science[sz-20:sz+20,sz-20:sz+20],interpolation='nearest', cmap=cm.gist_heat)
                plt.title('Science Image')
                plt.draw()
                plt.pause(0.00001)
                #print(",".join(["{0:4.1f}".format(a/self.dm_poke_scale) for a in coefficients_current]))
        
        return im_mn
        