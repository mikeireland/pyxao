# Using pyxao to compute the PSF of a very simple telescope. 
import pyxao
import opticstools as ot
import matplotlib
# Change the default interpolation behaviour
matplotlib.rc('image', interpolation='nearest')
import matplotlib.pyplot as plt
import numpy as np
import pdb

""" Variables """
dOut = 4.0			# outer annulus diameter (m)
dIn = 1.0			# inner annulus diameter (m)
sz = 1024			# side length (px)
mPerPix = 1.25 * dOut / sz	# pixel scale (m px^-1)
wavelength = 500e-9	# wavelength (m)

""" Defining a pupil for the wavefront """
wavefrontPupil = {	'type':'annulus',\
					'dout': dOut,\
					'din' : dIn\
					}

""" Plotting """
plt.close('all')
figCount = 0
# x and y limits to zoom in on the image plane.
denom = 30.0
xLower = (denom / 2.0 - 1)/denom * sz * 2
xUpper = (denom / 2.0 + 1)/denom * sz * 2
yLower = xLower
yUpper = xUpper

""" Calculating the PSF of the telescope """
myWavefrontPsf = pyxao.Wavefront(wavelength, mPerPix, sz, wavefrontPupil)
# At this point, the field is simply the default (uniform)
# We mask it with the pupil to get the E field at 
# the pupil plane.
myWavefrontPsf.field *= myWavefrontPsf.pupil
pupilFieldFlat = myWavefrontPsf.field
# Then, we call the image method to obtain the 
# field at the image plant corresponding to our field.
imagePlaneFieldFlat = 	myWavefrontPsf.image(True)
H = 					myWavefrontPsf.image(False)	# Note: the PSF (H) is intensity by definition

plt.figure(figCount)
figCount+=1


plt.subplot(221)
plt.imshow(pupilFieldFlat.real)
plt.title('Field at pupil plane (real)')
plt.colorbar()

plt.subplot(222)
plt.imshow(H)
plt.xlim(xLower, xUpper)
plt.ylim(yLower, yUpper)
plt.title('Telescope PSF (H)')
plt.colorbar()

plt.subplot(223)
plt.imshow(imagePlaneFieldFlat.real)
plt.xlim(xLower, xUpper)
plt.ylim(yLower, yUpper)
plt.title('Field at image plane (real)')
plt.colorbar()

plt.subplot(224)
plt.imshow(imagePlaneFieldFlat.imag)
plt.xlim(xLower, xUpper)
plt.ylim(yLower, yUpper)
plt.title('Field at image plane (imag)')
plt.colorbar()