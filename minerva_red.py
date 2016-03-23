from __future__ import print_function, division
import pyxao
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
#from PyQt4 import QtGui
#from PyQt4 import QtCore
#im_corrected
#im_bad_uncorrected
#im_bad_corrected
#im_uncorrected
#im_perfect

#Initialize some wavefronts. 0.56 to 0.76 microns sense (average 0.66 microns)
#0.77 to 0.93 microns science (average 0.85 microns). For such a narrow field of
#view, we can assume monochromatic 
pupil={'type':"annulus", 'dout':0.7,'din':0.32}
# Wavefront sensor wavelength
wf_sense = pyxao.Wavefront(wave=0.66e-6,m_per_pix=0.01,sz=128,pupil=pupil)
# Science wavelength
wf_image = pyxao.Wavefront(wave=0.85e-6,m_per_pix=0.01,sz=128,pupil=pupil)
# DM has both wavelengths reflected off it 
# dm  = pyxao.DeformableMirror(wavefronts=[wf_sense,wf_image],actuator_pitch=0.15,geometry='square', plotit=True)
# wfs = pyxao.ShackHartmann(wavefronts=[wf_sense],lenslet_pitch = 0.26,plotit=True)

dm  = pyxao.DeformableMirror(wavefronts=[wf_sense,wf_image],actuator_pitch=0.14,geometry='square', plotit=True)
wfs = pyxao.ShackHartmann(wavefronts=[wf_sense],lenslet_pitch = 0.11,plotit=True)

print("Click to continue!")
dummy = plt.ginput(1)

aos = pyxao.SCFeedBackAO(dm,wfs)
aos.find_response_matrix()
aos.compute_reconstructor()

#Add an atmosphere model to our wavefronts. 
atm = pyxao.Atmosphere(sz=1024, m_per_pix=wf_sense.m_per_pix) #For 1.7" seeing, try: ,r_0=[0.1,0.1,0.1])
wf_sense.add_atmosphere(atm)
wf_image.add_atmosphere(atm)

#See if the reconstructor works!
#sensors,ims  = aos.correct_twice()

im_mn = aos.run_loop(plotit=True)

#Uncomment the line below instead to see uncorrected seeing.
#im_mn = aos.run_loop(plotit=True,gain=0.0)
sz = im_mn.shape[0]
plt.imshow(im_mn[sz//2-20:sz//2+20,sz//2-20:sz//2+20],interpolation='nearest', cmap=cm.gist_heat)