from __future__ import division, print_function
import opticstools as ot
import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage as nd
import pdb
import time
import pdb
plt.ion()

try:
    import pyfftw
    pyfftw.interfaces.cache.enable()
    pyfftw.interfaces.cache.set_keepalive_time(1.0)
    nthreads=6 
except:
    nthreads=0


class Wavefront():
    """This is a base wavefront class. 
    
    It not only stores the wavefront, but 
    includes propagation and application of the telescope pupil function. As
    a start at least, there is only 1 wavelength. 
    
    Parameters
    ----------
    wave: float
        wavelength in meters
    m_per_pix: float
        meters per pixel
    sz: int
        size of the wavefront
    pupil: dict
        Pupil type and parameters."""
    def __init__(self,wave=1.65e-6,m_per_pix=0.02,sz=512,pupil={'type':"annulus", 'dout':8.0,'din':3.0}):
        self.wave=wave
        self.m_per_pix = m_per_pix
        self.sz = sz
        try:
            ptype = pupil['type']
        except:
            print("Pupil is a dictionary and must have a 'type' keyword")
            raise UserWarning
        if ptype == "annulus":
            self.pupil = ot.utils.circle(self.sz, pupil['dout']/self.m_per_pix) - ot.utils.circle(self.sz, pupil['din']/self.m_per_pix)
        
        else:
            print("Invalid pupil 'type' keyword!")
        #Initialize an empty list of propagators
        self.propagators = []
        #Initialize the wavefront field
        self.field = (1+0j)*np.ones( (sz,sz) )
        self.atmosphere_start = 0
        self.atm=None
        
    def add_propagator(self,distance, demag=1.0):
        """Add a propagator to the list of propagators
        
        Parameters
        ----------
        distance: float
            distance to propagate
        demag: float
            pupil demagnification"""
        self.propagators.append(ot.FresnelPropagator(self.sz,self.m_per_pix,distance*demag**2, self.wave))
        
    def propagate(self, index):
        """Propagate the wavefront, modifying the field appropriately.
        
        Parameters
        ----------
        index: int
            index of the propagator, from the list of propagators stored by this
            instance."""
        if len(self.propagators) <= index:
            print("ERRROR: index out of range")
            raise UserWarning
        self.field = self.propagators[index].propagate(self.field)
        
    def add_atmosphere(self,atm):
        """Add atmosphere propagators to this wavefront.
        
        Parameters
        ----------
        atm: Atmosphere instance
            The atmosphere instance for the propagators"""
        #Check atmosphere is compatible
        if (self.m_per_pix != atm.m_per_pix):
            print("ERROR: Atmosphere and wavefront must have the same scale")
        self.atmosphere_start = len(self.propagators)
        for dz in atm.dz:
            self.propagators.append(ot.FresnelPropagator(self.sz,self.m_per_pix,dz, self.wave))
        self.atm=atm
        
    def remove_atmosphere(self):
        """Remove atmosphere propagators and link to the atmosphere"""
        if not self.atm:
            print("ERROR: No atmosphere to remove!")
            raise UserWarning
        for i in range(self.atm.nlayers):
            self.propagators.pop(self.atmosphere_start)
        self.atm=None
            
    def pupil_field(self,edge_smooth=16):
        """Find the electric field at the telescope pupil, and place it in the field
        variable. It is assumed that the atmosphere is loaded into the first of 
        the propagators for this wavefront.
        
        Parameters
        ----------
        edge_smooth: int
            Smooth this many pixels from the edge when propagating through the atmosphere.
            Needed because the wavefront is *not* periodic after being truncated to the 
            size (sz) of this wavefront."""
        atm = self.atm
        if not atm:
            print("ERROR: Intitialise wavefront with add_atmosphere() first!")
            raise UserWarning
        #Check atmosphere is compatible
        if (self.m_per_pix != atm.m_per_pix):
            print("ERROR: Atmosphere and wavefront must have the same scale")
            raise UserWarning
        #Initialise the field.
        self.field = (1+0j)*np.ones( (self.sz,self.sz) )
        for i in range(atm.nlayers):
            #Apply the atmospheric phase. NB on a timing test, this is more time-intensive
            #than the Fresnel propagation.
            #tic = time.time()
            self.field *= np.exp(2j*np.pi*atm.delays[i][:self.sz,:self.sz]/self.wave)
            #Smooth the edges
            self.field[:edge_smooth,:] = 1 + (self.field[:edge_smooth,:]-1)* \
                np.repeat(np.arange(edge_smooth)/edge_smooth,self.sz).reshape(edge_smooth,self.sz)
            self.field[-edge_smooth:,:] = 1 + (self.field[-edge_smooth:,:]-1)* \
                np.repeat(np.arange(edge_smooth)[::-1]/edge_smooth,self.sz).reshape(edge_smooth,self.sz)
            self.field[:,:edge_smooth] = 1 + (self.field[:,:edge_smooth]-1)* \
                np.tile(np.arange(edge_smooth)/edge_smooth,self.sz).reshape(self.sz,edge_smooth)
            self.field[:,-edge_smooth:] = 1 + (self.field[:,-edge_smooth:]-1)* \
                np.tile(np.arange(edge_smooth)[::-1]/edge_smooth,self.sz).reshape(self.sz,edge_smooth)
            #toc = time.time()
            #Propagate to the next location
            self.field = self.propagators[i+self.atmosphere_start].propagate(self.field)
            #print("t1: {0:6.3f}, t2 {1:6.3f}".format(toc-tic, time.time()-toc))
        self.field *= self.pupil 
        
    def image(self, return_efield=False):
        """Return an image based on the current field. 
        
        Zero-pad in order to ensure nyquist sampling. Note that this is currently 
        super-flawed because it zero-pads the *same* amount for all wavelengths, so
        has a different image scale. Arguably, a better approach is to Fresnel propagate
        to a common focal plane.
        
        Parameters
        ----------
        return_efield: boolean
            Do we return an electric field? If not, return the intensity."""
        zpad = np.zeros( (self.sz*2,self.sz*2),dtype=np.complex)
        zpad[:self.sz,:self.sz]=self.field
        zpad = np.roll(np.roll(zpad,-self.sz//2,axis=1),-self.sz//2,axis=0)
        if (nthreads==0):
            efield = np.fft.fft2(zpad)
        else:
            efield = pyfftw.interfaces.numpy_fft.fft2(zpad,threads=nthreads)
        efield = np.fft.fftshift(efield)
        if return_efield:
            return efield
        else:
            return np.abs(efield)**2
            

            