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
sz = 512			# side length
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

# Getting the uncorrected pupil plane E field and image plane intensities
oldField = myWavefront.field
oldImage = myWavefront.image()

""" Deformable mirror """
myDm = pyxao.DeformableMirror(wavefronts=[myWavefront],plotit=False,geometry='hexagonal')
coeffs = np.random.random_sample(myDm.nactuators)*wavelength
# Applying wavefront correction
myDm.apply(coeffs)

# Getting the corrected pupil plane E field and image plane intensities
newField = myWavefront.field
newImage = myWavefront.image()

# Plotting
plt.figure(figCount)
# Uncorrected wavefront
plt.subplot(231)
plt.imshow(np.angle(oldField))
plt.title('WF phase before correction')
plt.colorbar()
# DM surface
plt.subplot(232)
plt.imshow(myDm.phasescreen/wavelength*2*np.pi)
plt.title('DM phase screen')
plt.colorbar()
# Corrected wavefront
plt.subplot(233)
plt.imshow(np.angle(newField))
plt.title('WF phase after correction')
plt.colorbar()
# Image plane intensity before correction
plt.subplot(234)
plt.imshow(oldImage)
plt.title('Image plane intensity before correction')
plt.colorbar()
# Image plane intensity after correction
plt.subplot(236)
plt.imshow(newImage)
plt.title('Image plane intensity after correction')
plt.colorbar()
