# Messing around with Mike's pyxao (and opticstools) code!
# Note: to use the pyxao module, we must be in the directly one above 
# the pyxao directory.
import pyxao
import opticstools as ot
import matplotlib
# Change the default interpolation behaviour
matplotlib.rc('image', interpolation='nearest')
import matplotlib.pyplot as plt
import numpy as np
import pdb

""" Playing with pyxao """
dOut = 4.0			# outer annulus diameter
dIn = 1.0			# inner annulus diameter
# Variables that must be shared between the wavefront and atmosphere instances:
sz = 1024			# side length
mPerPix = 1.25 * dOut / sz	# pixel scale
# Other variables
wavelength = 1064e-9	# wavelength
elevations = [10000.0]	# turbulent layer elevations
angle_wind = [0.0]		# angle of the wind (w.r.t. x-axis)
r0 = [50e-2]			# Fried parameter
airmass = [1.0]		# air mass
v_wind = [0.0]		# wind speed

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

""" Wavefront data """
# Propagators are stored in the propagators list.
# NOTE: there are two ways to add (Fresnel)propagators:
#	add_propagator(self, distance, demag)
#		Adds a propagator corresponding to the distance and demagnification given.
#		The actual distance of the propagation is given by distance * demag^2.
#	add_atmosphere(self, atm)
#		Adds propagators corresponding to the atmosphere object.
#		The actual distance of the propagation is given by atm.dz.
#		The variable atmosphere_start keeps track of where the atmosphere 
#		propagators begin in the list of propagators.

""" Image with atmospheric turbulence """
myWavefront = pyxao.Wavefront(wavelength, mPerPix, sz, wavefrontPupil)
print '======= Wavefront data ========'
print 'myWavefront.sz =', myWavefront.sz
print 'myWavefront.m_per_pix =', myWavefront.m_per_pix
# field is of type ndarray
# print 'field =', myWavefront.field
print 'size(myWavefront.field) =', myWavefront.field.size
print 'shape(myWavefront.field) =', myWavefront.field.shape
print 'myWavefront.wave =', myWavefront.wave
# print 'pupil =', myWavefront.pupil
print 'myWavefront.propagators =', myWavefront.propagators

""" Atmosphere data """
myAtmosphere = pyxao.Atmosphere(sz, mPerPix, elevations, v_wind, r0, angle_wind, airmass)
print '======= Atmosphere data ========'
print 'myAtmosphere.m_per_pix =', myAtmosphere.m_per_pix
print 'myAtmosphere.sz =', myAtmosphere.sz
# print 'myAtmosphere.elevations =', myAtmosphere.elevations
print 'myAtmosphere.dz =', myAtmosphere.dz
print 'myAtmosphere.v_wind =', myAtmosphere.v_wind
print 'myAtmosphere.angle_wind =', myAtmosphere.angle_wind
# print 'myAtmosphere.airmass =', myAtmosphere.airmass
print 'myAtmosphere.r_0 =', myAtmosphere.r_0

# DEMO code - broken (where are the phasescreens?)
# myAtmosphere.propagate_to_ground()

# Add an atmosphere
print '======= After adding atmosphere: ======='
myWavefront.add_atmosphere(myAtmosphere)
# Only one atmosphere can be added at a time! 
# After adding the atmosphere, we can call the pupil_field() method.
# pupil_field() computes the E field at the telescope pupil based on the atmosphere,
# and places it in the 'field' variable. 
# NOTE: pupil_field() doest NOT take into account propagators added by the add_propagator
# method, only those corresponding to the atmosphere object!
myWavefront.pupil_field()
pupilFieldTurb = myWavefront.field

# A propagator is added for each turbulent layer of the atmosphere.
# (and any others that are added through add_propagator) 
print 'myWavefront.propagators =', myWavefront.propagators

# Plotting
plt.figure(figCount)
figCount+=1

# Field at pupil plane
plt.subplot(121)
plt.imshow(pupilFieldTurb.real)
plt.title('Field at pupil plane (real)')
plt.colorbar()

# Field at image plane (real)
showField = True
plt.subplot(122)
# NOTE: image() shows the field at the image plane corresponding to
# the field variable. If pupil_field() has not been called then 
# field is still the default (1 + j at all points)
# The resulting field is simply the delta function! 
plt.imshow(myWavefront.image(showField).real)
plt.xlim(xLower, xUpper)
plt.ylim(yLower, yUpper)
plt.title('Field at image plane (real)')
plt.colorbar()
