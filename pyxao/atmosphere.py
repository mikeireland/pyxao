from __future__ import division, print_function
import opticstools as ot
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import scipy.ndimage as nd
import pdb

class Atmosphere():
    """A model of the atmosphere, including a fixed phase screen. 
    
    Note that the atmosphere has no knowledge
    of wavelength - it is the wavefronts that contain the monochromatic wavefront propagators.
    
    Parameters
    ----------
    sz: int
        Size (per side) of the atmosphere. NB should be *at least* 2 times larger than the 
        wavefront sensor size.
    m_per_px: float
        meters per pixel in the atmosphere. Must be the same as wavefronts - there is
        a !!!problem!!! with the code that this has to be set in both classes.
    elevations: list of floats
        Elevations of the turbulent layers in m
    v_wind: list of floats
        Velocities of the layers in m/s
    angle_wind: list of floats
        Angle of the wind directions in radians.
    airmass: float
        Airmass of observation (i.e. secant of zenith distance)
    r_0: list of floats
        Fried coherence lengths defined at 0.5 microns for each layer
    """
    def __init__(self,sz = 1024, 
        m_per_px=0.02,
        elevations=[10e3,5e3,0],
        v_wind=[20,10,5],
        r_0=[.2,.2,.2],
        angle_wind=[.1,.1,.1],
        airmass=1.0,
        wave_ref = .5e-6,
        seed=None
        ): 

        # Convert to lists if necessary
        if np.isscalar(elevations) and np.isscalar(v_wind) and np.isscalar(r_0) and np.isscalar(angle_wind):
            r_0 = [r_0]
            v_wind = [v_wind]
            elevations = [elevations]
            angle_wind = [angle_wind]

        self.nlayers=len(r_0)
        self.sz=sz
        self.m_per_px=m_per_px
        self.angle_wind = angle_wind
        self.time=0

        #Sanity check inputs
        if ( (len(elevations) != len(v_wind)) |
            (len(v_wind) != len(r_0)) |
            (len(r_0) != len(angle_wind)) ):
            pdb.set_trace()
            print("ERROR: elevations, v_wind, r_0 and angle_wind must all be the same length")
            raise UserWarning 
        
        #Correct layers for airmass
        elevations = np.append(elevations,0)
        self.dz = (elevations[1:]-elevations[:-1])*airmass
        self.r_0 = np.array(r_0)/airmass
        self.v_wind = np.array(v_wind)/airmass
        
        #Delays in meters at time=0
        self.delays0 = np.empty( (len(r_0),sz,sz) )
        for i in range(self.nlayers): 
            self.delays0[i] = ot.kmf(sz,seed) * np.sqrt(6.88*(m_per_px/r_0[i])**(5.0/3.0)) * wave_ref / 2 / np.pi
        
        #Delays in meters at another time.
        self.delays  = self.delays0.copy()
            
        
    def evolve(self, time=0):
        """Evolve the atmosphere to a new time
        
        Parameters
        ----------
        time: float
            Time at which to evolve the atmosphere (seconds)"""
        for i in range(self.nlayers):
            yshift_in_pix = self.v_wind[i]*time/self.m_per_px*np.sin(self.angle_wind[i])
            xshift_in_pix = self.v_wind[i]*time/self.m_per_px*np.cos(self.angle_wind[i])
            self.delays[i] = nd.interpolation.shift(self.delays0[i],(yshift_in_pix,xshift_in_pix),order=1,mode='wrap')
        
    def propagate_to_ground(self,wave=1e-6,dz=2e2,nprop=50):
        # AZ: dz is the distance over which to propagate the wavefront.
        """DEMO: Show a pretty movie of the full wavefront being propagated to ground level."""
        prop = ot.FresnelPropagator(self.sz,self.m_per_px, dz,wave)
        field = np.exp(2j*np.pi*self.phasescreens[0]/wave)
        for i in range(nprop):
            field = prop.propagate(field)
            plt.clf()
            hw = self.sz/2*self.m_per_px #half-width in metres
            plt.imshow(np.abs(field),cmap=cm.gray,interpolation='nearest',vmax=2,\
                extent=[-hw,hw,-hw,hw])
            plt.title("Field amplitude after {0:5.1f}m. RMS: {1:5.2f}".format((i+1)*dz, np.std(np.abs(field))))
            plt.draw()
            
        
