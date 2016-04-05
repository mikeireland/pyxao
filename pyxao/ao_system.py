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

        print("Sensing wavefronts:")
        print(image_ixs)

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
        
    def run_loop(self,
                dt=0.002,
                nphot=1e4,
                mode='PID',
                niter=1000,
                nframesbetweenplots=10,
                plotit=False,
                K_p=0.0,
                K_i=1.0,
                K_d=0.0,
                K_leak=0.9):
        """Run an AO servo loop.
        
        It is assumed that all wavefronts share the same atmosphere!
        
        Parameters
        ----------
        dt: float
            Time between samples
        nphot: float
            Number of photons per frame (noise)
        mode: string
            Servo loop mode. 'PID' is simple proportional-integral-derivative control. 
            For PD, P, I etc. control, simply set the unneeded gain terms to zero.
        niter: int
            Number of iterations of the loop.
        plotit: boolean
            Do we plot potentially useful outputs? NB, this is all pretty slow with
            matplotlib.
        K_i: float
            Integral gain. 
        K_leak: float
            Leak coefficient for integral control. (allows transient errors in WFS 
            measurements to leak out of the control input over time)
            Is automatically zeroed if K_i is zeroed. 
        """
        # Zero the leak coefficient if no integral control is used. 
        if K_i == 0.0:
            K_leak = 0.0

        # DM coefficients (control commands)
        coefficients_current = np.zeros(self.dm.nactuators) # timestep k
        
        # WFS measurements
        y_current = np.zeros(self.wfs.nsense)    # timestep k
        y_old = np.zeros(self.wfs.nsense)        # timestep k - 1

        sz = self.dm.wavefronts[-1].sz
        image_mean = np.zeros( (2*sz,2*sz) )
        im_science = np.zeros( (len(self.image_ixs),sz*2,sz*2) )
        im_science_all = np.zeros( (sz*2,sz*2) )

        # nims = 0
        im_perfect = np.zeros( (sz*2,sz*2) )
        for ix in self.image_ixs:
            self.dm.wavefronts[ix].field = self.dm.wavefronts[ix].pupil
            im_perfect += self.dm.wavefronts[ix].image()   

        """ AO Control loop """
        for k in range(niter):
            # Evolve the atmosphere.
            self.dm.wavefronts[0].atm.evolve(dt*k)
            
            # Reset the pupil fields.
            for wf in self.dm.wavefronts:
                wf.pupil_field()
            
            # Apply the wavefront correction.
            self.dm.apply(coefficients_current)
            
            # Sense the corrected wavefront.
            sensors = self.wfs.sense(nphot=nphot)
            y_old = y_current               # y at timestep k - 1
            y_current = sensors.flatten()   # y at timestep k
            
            # Apply control logic.
            #TODO: check that these work!
            coefficients_next = np.zeros(self.dm.nactuators)    # timestep k + 1
            if mode=='PID':
                coefficients_next += K_leak * coefficients_current - K_i * np.dot(self.reconstructor,y_current) * self.dm_poke_scale
                coefficients_current += - K_p * np.dot(self.reconstructor,y_current) * self.dm_poke_scale
                coefficients_current += - K_d * (1/dt) * (np.dot(self.reconstructor,y_current) - np.dot(self.reconstructor,y_old)) * self.dm_poke_scale
            else:
                print('ERROR: invalid control loop mode! Use PID for now...')
                raise UserWarning

            coefficients_current = coefficients_next;

            # Create the image. By FFT for now...
            im_science = np.zeros( (len(self.image_ixs),sz*2,sz*2) )
            im_science_all = np.zeros( (sz*2,sz*2) )

            # Remember that image_ixs are indices in the DM wavefront list, 
            # NOT the WFS wavefront list.
            im_ix = 0
            for ix in self.image_ixs:
                # Huygens propagation to generate the science image
                im_science_all += self.dm.wavefronts[ix].image()
                im_science[im_ix] = self.dm.wavefronts[ix].image()
                im_ix += 1

            if (k != 0):
                image_mean += im_science_all 
                # nims += 1
                     
            if plotit & ((k % nframesbetweenplots)==0):  
                if k == 0:
                    plt.rc('text', usetex=True)
                    fig = plt.figure()

                    ax1 = fig.add_subplot(231)  # WFS detector image
                    ax1.title.set_text(r'WFS detector')
                    ax1.axis( [0,self.dm.wavefronts[0].sz,0,self.dm.wavefronts[0].sz] )
                    ax2 = fig.add_subplot(232)  # Corrected phase 1
                    ax2.title.set_text(r'Corrected phase, $\lambda$ = %d nm' % (self.dm.wavefronts[self.image_ixs[0]].wave*1e9))
                    ax4 = fig.add_subplot(233)  # Science image 1
                    ax4.title.set_text(r'Science Image ($\lambda$ = %d nm)' % (self.dm.wavefronts[self.image_ixs[0]].wave*1e9))
                    plot1 = ax1.imshow(self.wfs.im,interpolation='nearest',cmap=cm.gray)
                    plot2 = ax2.imshow(np.angle(self.dm.wavefronts[self.image_ixs[0]].field)*self.dm.wavefronts[self.image_ixs[0]].pupil,interpolation='nearest', cmap=cm.gist_rainbow)
                    plot4 = ax4.imshow(im_science[0,sz-20:sz+20,sz-20:sz+20],interpolation='nearest', cmap=cm.gist_heat)

                    if len(self.image_ixs) > 1:
                        ax3 = fig.add_subplot(235)  # Corrected phase 2
                        ax3.title.set_text(r'Corrected phase, $\lambda$ = %d nm' % (self.dm.wavefronts[self.image_ixs[1]].wave*1e9))
                        ax5 = fig.add_subplot(236)  # Science image 2
                        ax5.title.set_text(r'Science Image ($\lambda$ = %d nm)' % (self.dm.wavefronts[self.image_ixs[1]].wave*1e9))
                        plot3 = ax3.imshow(np.angle(self.dm.wavefronts[self.image_ixs[0]].field)*self.dm.wavefronts[self.image_ixs[1]].pupil,interpolation='nearest', cmap=cm.gist_rainbow)
                        plot5 = ax5.imshow(im_science[1,sz-20:sz+20,sz-20:sz+20],interpolation='nearest', cmap=cm.gist_heat)
                    
                else:
                    plot1.set_data(self.wfs.im)
                    plot2.set_data(np.angle(self.dm.wavefronts[self.image_ixs[0]].field)*self.dm.wavefronts[self.image_ixs[0]].pupil)
                    plot4.set_data(im_science[0,sz-20:sz+20,sz-20:sz+20])
                    if len(self.image_ixs) > 1:
                        plot3.set_data(np.angle(self.dm.wavefronts[self.image_ixs[1]].field)*self.dm.wavefronts[self.image_ixs[1]].pupil)
                        plot5.set_data(im_science[1,sz-20:sz+20,sz-20:sz+20])

                plt.draw()
                plt.pause(0.00001)

        image_mean /= niter

        #TODO: plot Strehl as a function of time 

        return image_mean, im_perfect
        