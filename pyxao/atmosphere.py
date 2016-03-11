from __future__ import division, print_function
import opticstools as ot
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pdb
plt.ion()

class Atmosphere():
    """This is the atmosphere class. """
    def __init__(self,sz = 512, m_per_pix=0.005,layers={'elevations':[10e3,0],'v_wind':[20,5],'r_0':[0.1,.1]},waves=[.5e-6]):
        """
        r_0: Defined at 0.5 microns.
        """
        wave_ref = .5e-6 #reference for r_0
        self.layers=layers
        self.sz=sz
        self.m_per_pix=m_per_pix
        self.waves=waves
        self.phasescreens=[]
        for r_0 in layers['r_0']: #One wavelength only! Set to 0.5 microns!!!
            self.phasescreens.append(ot.kmf(sz) * np.sqrt(6.88*(m_per_pix/r_0)**(5.0/3.0)) * wave_ref/waves[0])
            
        #Create the propagators
        self.propagators = []
        el = layers['elevations']
        for i in range(len(el) - 1):
            #Propagate between two layers, one pair of layers at a time.
            self.propagators.append(ot.FresnelPropagator(self.sz,self.m_per_pix, el[i] - el[i+1],self.waves[0]))
        
        
    def pupil_field(self,time=0):
        """Find the electric field at the telescope pupil"""
        
        #!!! NB no interpolation here yet. Does this matter?
        yshift_in_pix = int(self.layers['v_wind'][0]*time/self.m_per_pix)
        field =    np.exp(2j*np.pi*np.roll(self.phasescreens[0], yshift_in_pix,axis=0))
        for i in range(1,len(self.phasescreens)):
            field = self.propagators[i-1].propagate(field)
            yshift_in_pix = int(self.layers['v_wind'][i]*time/self.m_per_pix)
            field *= np.exp(1j*np.roll(self.phasescreens[i],yshift_in_pix,axis=0))
        return field   
        
    def propagate_to_ground(self,dz=2e2,nprop=50):
        """Show a pretty movie of the full wavefront being propagated to ground level"""
        prop = ot.FresnelPropagator(self.sz,self.m_per_pix, dz,self.waves[0])
        field = np.exp(1j*self.phasescreens[0])
        for i in range(nprop):
            field = prop.propagate(field)
            plt.clf()
            hw = self.sz/2*self.m_per_pix #half-width in metres
            plt.imshow(np.abs(field),cmap=cm.gray,interpolation='nearest',vmax=2,\
                extent=[-hw,hw,-hw,hw])
            plt.title("Field amplitude after {0:5.1f}m. RMS: {1:5.2f}".format((i+1)*dz, np.std(np.abs(field))))
            plt.draw()
            
        
        
# Atmosphere.dx
# Atmosphere.layers["v_wind"]
# Atmosphere.layers["theta_wind"]
# Atmosphere.layers["r_0"]
# Atmosphere.layers["lambda_ref"]
# Atmosphere.layers["evolution_time"]
# Atmosphere.layers["evolution_type"] 
# Atmosphere.apply(wavefront, dt, lambda)
# 
# #For GMT: A special function that takes out some fraction of piston, leaving the rest behind.
# Atmosphere.apply(wavefront,dt, lambda) 
