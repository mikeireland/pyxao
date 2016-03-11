from __future__ import division, print_function
import opticstools as ot
import numpy as np
import matplotlib.pyplot as plt
import pdb
plt.ion()

class Wavefront():
    """This is a base wavefront class. It not only stores the wavefront, but 
    includes propagation and application of the telescope pupil function. As
    a start at least, there is only 1 wavelength. """
    def __init__(self,wave=1e-6,m_per_pix=0.02,sz=512,pupil={'type':"annulus", 'dout':8.0,'din':3.0}):
        self.wave=wave
        self.m_per_pix = m_per_pix
        self.sz = sz
        try:
            ptype = pupil['type']
        except:
            print("Pupil is a dictionary and must have a 'type' keyword")
            raise UserWarning
        if ptype == "annulus":
        
        else:
            print("Invalid pupil 'type' keyword!")
        #Initialize an empty list of propagators
        self.propagators = []
        #Initialize the wavefront field
        self.field = (1+0j)*np.ones( (sz,sz) )
        
    def add_propagator(self,distance, magnification=1.0):
        """Add a propagator to the list of propagators"""
        self.propagators.append(ot.FresnelPropagator(self.sz,self.m_per_pix,distance*magnification**2, self.wave))
        