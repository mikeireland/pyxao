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
wf_sense = pyxao.Wavefront(wave=0.66e-6,m_per_px=0.01,sz=256,pupil=pupil)
# Science wavelength
wf_image = pyxao.Wavefront(wave=0.85e-6,m_per_px=0.01,sz=256,pupil=pupil)

dm  = pyxao.DeformableMirror(wavefronts=[wf_sense,wf_image],actuator_pitch=0.115,geometry='square', plotit=False,central_actuator=True)
wfs = pyxao.ShackHartmann(wavefronts=[wf_sense],lenslet_pitch = 0.105,central_lenslet=False,sampling=1,plotit=False,geometry='hexagonal',N_phot = 1600 * 1e4 * 1/2000 * 0.90 * 1000)

#Add an atmosphere model to our wavefronts. 
atm = pyxao.Atmosphere(sz=1024, m_per_px=wf_sense.m_per_px,r_0=[0.1,0.1,0.1]) #For 1.7" seeing, try: ,r_0=[0.1,0.1,0.1])

aos = pyxao.SCFeedBackAO(dm,wfs,atm,image_ixs=1)
aos.find_response_matrix()
aos.compute_reconstructor(threshold=0.1)

# Want 1600 photons/cm2/s for the LGS. 
# = photons/m/s = 1600 * 1e4
# photons in im = 1600 * 1e4 * D * 1/2000

#See if the reconstructor works!
#sensors,ims  = aos.correct_twice()

# pdb.set_trace()
psfs = aos.run_loop(plotit=True,dt=0.002,niter=50,mode='integrator',plate_scale_as_px=0.1,psf_ix = 1,nframesbetweenplots=10)
psf_mn = np.mean(psfs,axis=0)
#Uncomment the line below instead to see uncorrected seeing.
# im_mn = aos.run_loop(plotit=True,dt=0.002,nphot=1e4,niter=50,mode='dodgy_damping',dodgy_damping=0.0,gain=0.0,plate_scale_as_px=0.125)[-2]

#Strehl: Regrid in order to sample finely the peak.
psf_dl = aos.psf_dl(plate_scale_as_px=0.1,psf_ix=1)
strehl = np.max(ot.utils.regrid_fft(psf_mn,(1024,1024)))/np.max(ot.utils.regrid_fft(psf_dl,(1024,1024)))

sz = psf_mn.shape[0]
plt.figure()
plt.imshow(psf_mn[sz//2-20:sz//2+20,sz//2-20:sz//2+20],interpolation='nearest', cmap=cm.gist_heat)
print("Strehl: {0:6.3f}".format(strehl))
