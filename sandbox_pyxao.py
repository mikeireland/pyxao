# A sandbox for playing with the opticstools, pyxao and linguini-sim packages.
# Goal: want to get an AO-corrected, turbulence-added image from some truth image (i.e. diffraction-limited image of a satellite) assuming the AO system and telescope parameters on the 1.8 m telescope.

# The steps:
# 1. Upload a truth image.
# 2. Generate a time series of PSFs using an appropriate AO system and turbulent layers.
# 3. Convolve each PSF with the truth image.

# To begin with:
# Make an AO system and atmosphere and try to get the PSFs out. 
# Go through the code and make sure things like plate scales, FFT padding etc. are correct.

# Things to keep in mind:
# - if it's too difficult or ends up taking too long then revert to OOMAO.
# - DON'T GET DISTRACTED BY SHINY
# - Fresnel propagation is more accurate but is it necessary here? Is it still correct?

#########################################################################################
from __future__ import division
import opticstools
import pyxao
import apdsim

# Telescope parameters
D_out = 1.8		# Telescope primary mirror diameter
D_in = 0.250	# Telescope secondary mirror diameter (central obscuration)
wavefrontPupil = {	
	'type':'annulus',
	'dout': D_out,
	'din' : D_in
}

# AO system parameters
wavelength_lgs_m = 589e-9
wavelength_science_m = FILTER_BANDS_M['K'][0]
# Seeing conditions
r0_500nm = 5e-2		# Corresponds to 'good seeing' at 500 nm. Used here to calculate r0 at other wavelengths
r0_lgs = [np.power((wavelength_lgs_m / 500e-9), 1.2) * r0_500nm]
r0_science = [np.power((wavelength_science_m / 500e-9), 1.2) * r0_500nm]

# Detector parameters
height_px = 320					# Detector side length (pixels)
m_per_px = D_out / height_px	# Physical mapping of detector onto primary mirror size

""" Wavefronts """
wf_wfs = pyxao.Wavefront(
	wave = wavelength_lgs_m,
	m_per_px = m_per_px,
	sz = height_px,
	puil = wavefrontPupil
	)
