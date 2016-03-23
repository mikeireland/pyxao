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

""" Wavefronts """
# Pupil
dOut = 2.0			# outer annulus diameter
dIn = 0.4 			# inner annulus diameter
wavefrontPupil = {	'type':'annulus',\
					'dout': dOut,\
					'din' : dIn\
					}
# Pixel scale
sz = 512			# side length
mPerPix = 1.25 * dOut / sz	# pixel scale
# Optical properties
wavelength = 1064e-9	# wavelength

wf = pyxao.wavefront(\
	wavelength,\
	mPerPix,\
	sz,\
	wavefrontPupil\
	)

""" Atmosphere """
# Atmospheric conditions 
elevations = [10000.0]	# turbulent layer elevations
angleWind = [0.0]	# angle of the wind (w.r.t. x-axis)
r0 = [10e-2]		# Fried parameter
airmass = [1.0]		# air mass
vWind = [0.0]		# wind speed     

atm = pyxao.Atmosphere(\
	sz,\
	mPerPix,\
	elevations,\
	vWind,\
	r0,\
	angleWind,\
	airmass\
	)


