from __future__ import division, print_function
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
plt.ion()

from wfs import *
from atmosphere import *
from wavefront import *
from deformable_mirror import *

#These below should go in a demo program...

def show_seeing(dt=0.02,nt=51,wave=1e-6):
    """DEMO: Show the intensity and phase in the pupil plane, for a default
    atmosphere and a default wavefront"""
    print("Initializing")
    atm = Atmosphere()
    wf = Wavefront(wave=wave)
    wf.add_atmosphere(atm)
    print("Running")
    for i in range(nt):
        tic1=time.time()
        atm.evolve(dt*i)
        tic2=time.time()
        wf.pupil_field(atm)
        tic3=time.time()
        im = wf.image()
        tic4=time.time()
        print("Times: {0:5.2f} {1:5.2f} {2:5.2f}".format(tic2-tic1,tic3-tic2,tic4-tic3))
        im /= np.max(im)
        plt.clf()
        hw = wf.sz/2*wf.m_per_pix #half-width in metres
        plt.subplot(221)
        plt.imshow(np.abs(wf.field),cmap=cm.gray,extent=[-hw,hw,-hw,hw],vmax=2,interpolation='nearest')
        plt.title("Amplitude")
        plt.subplot(222)
        plt.imshow(np.angle(wf.field)*wf.pupil,extent=[-hw,hw,-hw,hw],vmax=2,interpolation='nearest')
        plt.title("Phase (T={0:6.2f})".format(dt*i))
        plt.subplot(223)
        hw_pix = int(np.radians(1./3600)*wf.sz*wf.m_per_pix/wf.wave*2)
        plt.imshow(im[wf.sz-hw_pix:wf.sz+hw_pix,wf.sz-hw_pix:wf.sz+hw_pix],interpolation='nearest', cmap=cm.gist_heat,extent=[-1,1,-1,1])
        plt.title("Image")
        
        #Now find a perfectly phase-corrected image
        wf.field = np.abs(wf.field)
        im_corrected =wf.image()
        im_corrected /= np.max(im_corrected)
        plt.subplot(224)
        hw_pix = int(np.radians(1./3600)*wf.sz*wf.m_per_pix/wf.wave*2)
        plt.imshow(np.arcsinh(np.minimum(im_corrected[wf.sz-hw_pix:wf.sz+hw_pix,wf.sz-hw_pix:wf.sz+hw_pix],.1)/1e-6),\
            interpolation='nearest', cmap=cm.gist_heat,extent=[-1,1,-1,1])
        plt.title("Corrected Image")
        plt.draw()