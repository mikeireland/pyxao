# Messing around with Mike's pyxao (and opticstools) code!
# Note: to use the pyxao module, we must be in the directory one above 
# the pyxao directory.
import pyxao
import opticstools as ot
import matplotlib
# Change the default interpolation behaviour
matplotlib.rc('image', interpolation='nearest')
import matplotlib.pyplot as plt
import numpy as np
import pdb

plt.close('all')
figCount = 0

""" Variables """
dOut = 2.0			# outer annulus diameter
dIn = 0.4 			# inner annulus diameter
sz = 256			# side length
mPerPix = 1.25 * dOut / sz	# pixel scale
wavelength = 1064e-9	# wavelength
elevations = [10000.0]	# turbulent layer elevations
angle_wind = [0.0]	# angle of the wind (w.r.t. x-axis)
r0 = [10e-2]		# Fried parameter
airmass = [1.0]		# air mass
v_wind = [0.0]		# wind speed       
wavefrontPupil = {	'type':'annulus',\
					'dout': dOut,\
					'din' : dIn\
					}

""" Wavefront & atmosphere """
myWavefront = pyxao.Wavefront(wavelength, mPerPix, sz, wavefrontPupil)
myAtmosphere = pyxao.Atmosphere(sz, mPerPix, elevations, v_wind, r0, angle_wind, airmass)
myWavefront.add_atmosphere(myAtmosphere)
myWavefront.pupil_field()
plt.imshow(myWavefront.field.real)
# pdb.set_trace()
myWavefront.phase = np.angle(myWavefront.field)

""" Basic wavefront sensor """
# myWfs = pyxao.WFS([myWavefront], myWavefront.pupil)
myWfs = pyxao.WFS([myWavefront])
# plt.figure(figCount)
# figCount += 1
# plt.imshow(myWfs.sense())

""" Shack-Hartmann wavefront sensor """
pxPerSubap = 12
lensletPitch = pxPerSubap * myWavefront.m_per_pix
myShWfs = pyxao.ShackHartmann(\
		wavefronts=[myWavefront],\
	 	geometry='square',\
	 	lenslet_pitch=lensletPitch\
	 )

# NOTE: upon creation of the SH WFS, sense() is called.
# Even if the wavefronts passed into the constructor have been 
# propagated earlier, the wavefront is flattened (see line 158 in wfs.py).
# To obtain the image corresponding to the crumpled wavefront, pupil_field()
# and sense() must both be called again.  

myWavefront.pupil_field()
myShWfs.sense()
# pdb.set_trace()

# Note: the sampling parameter doesn't do anything yet
plt.figure(figCount)
figCount += 1
plt.subplot(121)
plt.imshow(myWavefront.phase)
plt.title('Wavefront phase')
plt.colorbar()
plt.subplot(122)
plt.imshow(myShWfs.im)
plt.title('SH WFS detector image')
plt.colorbar()

