from __future__ import division, print_function
import opticstools as ot
import numpy as np
import matplotlib.pyplot as plt
import pdb
import scipy.ndimage as nd
import scipy.linalg as la
import matplotlib.cm as cm
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
        location of wfs and DM conguagate
    image_ixs: list
        A list of indexes for which images should be calculated, i.e. the 
        science wavefront indices from the dm instance.
    dm_poke_scale: float
        A normalisation for poking the deformable mirror, used in response matrix
        sensing and the loop. Should be well within the linear regime."""
    def __init__(self,dm,wfs,conjugate_location=0.0,image_ixs=None,
        dm_poke_scale=1e-7):
        
        if not image_ixs:
            image_ixs = range(len(dm.wavefronts),len(wfs.wavefronts))
        if conjugate_location != 0:
            print("OOPS: Not implemented yet - only ground layer conugation so far")
            raise UserWarning
            
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
            for i in range(self.dm.nactuators):
                #Flatten the WFS wavefronts
                for wf in self.wfs.wavefronts:
                    wf.field=wf.pupil
                act = np.zeros(self.dm.nactuators)
                act[i] = self.dm_poke_scale
                self.dm.apply(act)
                wfs_plus = self.wfs.sense() 
                for wf in self.wfs.wavefronts:
                    wf.field=wf.pupil
                act = np.zeros(self.dm.nactuators)
                act[i] = -self.dm_poke_scale
                self.dm.apply(act)
                wfs_minus = self.wfs.sense() 
                #Check that poking in both directions is equivalent!
                if (np.sum(wfs_plus*wfs_minus)/np.sum(wfs_plus*wfs_plus) > -0.9):
                    print("WARNING: Poking the DM is assymetric!!!")
                    pdb.set_trace()
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
            
    def correct_twice(self):
        """Find the pupil field, then correct it twice.
        
        TEST method, but ERROR checking still needed!
        
        Returns
        -------
        sensors: list
            A list of the sensor output for before correction, after 1 and after 2
            applications of the reconstructor
        ims: list
            A list of wavefront sensor images (before correction, after 1 and 2
            reconstructor applications)
        """
        for wf in self.wfs.wavefronts:
            wf.pupil_field()
        #Sense the wavefront.
        sensors0 = self.wfs.sense()
        im0 = self.wfs.im.copy()
        actuators = -np.dot(self.reconstructor,sensors0.flatten())*self.dm_poke_scale
        for wf in self.wfs.wavefronts:
            wf.pupil_field()
        self.dm.apply(actuators)
        sensors1 = self.wfs.sense()
        im1 = self.wfs.im.copy()
        actuators += -np.dot(self.reconstructor,sensors1.flatten())*self.dm_poke_scale
        for wf in self.wfs.wavefronts:
            wf.pupil_field()
        self.dm.apply(actuators)
        sensors2 = self.wfs.sense()
        im2 = self.wfs.im.copy()
        return [sensors0,sensors1,sensors2],[im0,im1,im2] 
        
    def run_loop(self,dt=0.002,nphot=1e4,mode='integrator',niter=1000,plotit=False,\
        gain=1.0,dodgy_damping=0.9):
        """Run an AO servo loop.
        
        It is assumed that all wavefronts share the same atmosphere!
        
        Parameters
        ----------
        dt: float
            Time between samples
        nphot: float
            Number of photons per frame
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
        actuators_current = np.zeros(self.dm.nactuators)
        sz = self.dm.wavefronts[-1].sz
        im_mn = np.zeros( (2*sz,2*sz) )
        for i in range(niter):
            self.dm.wavefronts[0].atm.evolve(dt*i)
            #Create the pupil fields.
            for wf in self.dm.wavefronts:
                wf.pupil_field()
            self.dm.apply(actuators_current)
            
            #Save a copy of the corrected pupil phase
            corrected_field = self.dm.wavefronts[0].field.copy()
            
            #Sense the wavefront
            sensors = self.wfs.sense(nphot=nphot)
            actuators_current = dodgy_damping*actuators_current - \
                gain*np.dot(self.reconstructor,sensors.flatten())*self.dm_poke_scale
            
            #Create the image. By FFT for now...
            im_science = np.zeros( (sz*2,sz*2) )
            for ix in self.image_ixs:
                im_science += self.dm.wavefronts[ix].image()
            im_mn += im_science 
            
            #Plot stuff if we want.
            if plotit & ((i % 10)==0):
                plt.clf()
                plt.subplot(311)
                plt.imshow(self.wfs.im,interpolation='nearest',cmap=cm.gray)
                plt.plot(self.wfs.px[:,0], self.wfs.px[:,1],'x')
                plt.axis( [0,self.dm.wavefronts[0].sz,0,self.dm.wavefronts[0].sz] )
                plt.title('WFS')
                plt.subplot(312)
                plt.imshow(np.angle(corrected_field)*self.dm.wavefronts[0].pupil,interpolation='nearest')
                plt.title('Corrected Phase')
                plt.subplot(313)
                plt.imshow(im_science[sz-20:sz+20,sz-20:sz+20],interpolation='nearest', cmap=cm.gist_heat)
                plt.title('Science Image')
                plt.draw()
                #print(",".join(["{0:4.1f}".format(a/self.dm_poke_scale) for a in actuators_current]))
        
        return im_mn
        