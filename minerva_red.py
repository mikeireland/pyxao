from __future__ import print_function, division
import pyxao
import opticstools as ot
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pdb
plt.ion()
np.random.seed(1)
#from PyQt4 import QtGui
#from PyQt4 import QtCore
#Strehl:
#np.max(ot.utils.regrid_fft(im_mn_on,(1024,1024)))/np.max(ot.utils.regrid_fft(im_mn,(1024,1024)))

#Initialize some wavefronts. 0.56 to 0.76 microns sense (average 0.66 microns)
#0.77 to 0.93 microns science (average 0.85 microns). For such a narrow field of
#view, we can assume monochromatic 
pupil={'type':"annulus", 'dout':0.7,'din':0.32}
# Wavefront sensor wavelength
wf_sense = pyxao.Wavefront(wave=0.66e-6,m_per_px=0.01,sz=128,pupil=pupil)
# Science wavelength
wf_image = pyxao.Wavefront(wave=0.85e-6,m_per_px=0.01,sz=128,pupil=pupil)

#dm  = pyxao.DeformableMirror(wavefronts=[wf_sense,wf_image],actuator_pitch=0.15,geometry='square', plotIt=True)
#wfs = pyxao.ShackHartmann(wavefronts=[wf_sense],lenslet_pitch = 0.26,plotIt=True)

dm  = pyxao.DeformableMirror(wavefronts=[wf_sense,wf_image],actuator_pitch=0.115,geometry='square', plotIt=False,central_actuator=True)
wfs = pyxao.ShackHartmann(wavefronts=[wf_sense],lenslet_pitch = 0.105,sampling=1,plotIt=False,geometry='hexagonal')

#pdb.set_trace()
#print("Click to continue")
#dummy=plt.ginput(1)

aos = pyxao.SCFeedBackAO(dm,wfs)
aos.find_response_matrix()
aos.compute_reconstructor(threshold=0.1)

#Add an atmosphere model to our wavefronts. 
atm = pyxao.Atmosphere(sz=1024, m_per_px=wf_sense.m_per_px,r_0=[0.1,0.1,0.1]) #For 1.7" seeing, try: ,r_0=[0.1,0.1,0.1])
wf_sense.add_atmosphere(atm)
wf_image.add_atmosphere(atm)

#See if the reconstructor works!
#sensors,ims  = aos.correct_twice()
# pdb.set_trace()
im_mn, im_perfect = aos.run_loop(plotIt=False,dt=0.002,nphot=1e4,niter=50,mode='dodgy_damping',dodgy_damping=0.95,gain=1.0,plate_scale_as_px=0.1)[-2:]
#Uncomment the line below instead to see uncorrected seeing.
# im_mn = aos.run_loop(plotIt=True,dt=0.002,nphot=1e4,niter=50,mode='dodgy_damping',dodgy_damping=0.0,gain=0.0,plate_scale_as_px=0.125)[-2]

#Strehl: Regrid in order to sample finely the peak.
# strehl = np.max(ot.utils.regrid_fft(im_mn,(1024,1024)))/np.max(ot.utils.regrid_fft(im_perfect,(1024,1024)))

sz = im_mn.shape[0]
plt.figure()
plt.imshow(im_mn[sz//2-20:sz//2+20,sz//2-20:sz//2+20],interpolation='nearest', cmap=cm.gist_heat)
# print("Strehl: {0:6.3f}".format(strehl))
