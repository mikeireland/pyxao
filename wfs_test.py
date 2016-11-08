# Adding a new feature: enabling the wavefront sensor detector to be sampled 
# independently of the wavefronts' pixel resolution
from __future__ import division, print_function
from aosim.pyxao import wfs, atmosphere, wavefront
import numpy as np
import ipdb

wave_height_px = 256

# Wavefronts
pupil={'type':"annulus", 'dout':0.7,'din':0.32}
wf = wavefront.Wavefront(wave=850e-9,
	m_per_px=0.7/256,
	pupil=pupil,
	sz=wave_height_px)

# wf_old = wavefront.Wavefront(wave=850e-9,
# 	m_per_px=0.7/105,
# 	pupil=pupil,
# 	sz=105)

# Atmosphere. Keep default values for now
atm = atmosphere.Atmosphere(m_per_px=0.7/256)
atm_old = atmosphere.Atmosphere(m_per_px=0.7/105)

# Wavefront sensor. Keep it square for now
# wfs_old = wfs.OldShackHartmann(
# 	wavefronts=[wf],
# 	lenslet_pitch_m = 0.7/7,
# 	N_os = 1,
# 	geometry = 'square',
# 	central_lenslet=True,
# 	plotit=True
# 	)

wfs = wfs.ShackHartmann(
	wavefronts=[wf],
	plate_scale_as_px = 3600 * np.rad2deg(850e-9/pupil['dout']/2),
	detector_sz = 105,
	nlenslets_per_side=7,
	geometry = 'square',
	plotit=True
	)


# In its current state, it seems that the number of pixels per lenslet is set by the wavefront's resolution. 
# We should be able to set it independently instead!
# Input arguments:
#	- detector size (separate from wavefront size)
#	- number of lenslets per size
#	- focal length (scaled up to aperture size?)
#		- Better to input the F# (= f/D)
#	- OR Nyquist sampling N_os
# Formula for Nyquist sampling:
#	N_os = 1 ==> 2 pixels per FWHM, so
#	plate_scale_as_px = lambda / 2 / D
# The pupil mask is calculated using the focal length, so 
# we need to be able to convert N_os into focal length
# Need to make sure it's not possible to input conflicting parameters - 
# for now stick with N_os, detector size and # lenslets 

# OK - given N_os, detector_sz and nlenslets, how do we calculate the focal 
# length/F#?
# plate_scale_as_px from N_os at the given wavelength
# (note: will have to specify Nyquist sampling AT a wavelength! so might be 
# better to put in plate scale instead)
# plate_scale_as_px = lambda / 2 / D / N_os, where
# D = pupil['dout'] / nlenslets 
# Ok, so how can we compute the fratio from this?
# plate_scale_as_mm = 206265/focal length in mm
# plate_scale_as_mm = plate_scale_as_px/l_px
# where l_px = pupil['dout']/detector_sz
# So fratio = 

# Note: all of these dimensions have been scaled up to the size of the primary
# aperture.

# To test: initialise the WFS, flatten the field and sense!

