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

def printDivider():
	print '======================================'
	return

def newFigure():
	plt.figure(newFigure.figCount)
	newFigure.figCount += 1
newFigure.figCount = 0

# Find the PSF of the AO system at a given wavelength.
def findPSF(wf, plotit=False):
	myWavefrontPsf = pyxao.Wavefront(
		wave=wf.wave, 
		m_per_pix=wf.m_per_pix, 
		sz=wf.sz)
	myWavefrontPsf.pupil = wf.pupil
	# After initialisation, the field is uniform - corresponds to
	# a point source at infinity.
	# Now, we need to mask it by the pupil:
	myWavefrontPsf.field *= myWavefrontPsf.pupil
	# The PSF is the resulting intensity at the image plane.
	psf = myWavefrontPsf.image(return_efield=False)
	if plotit:
		axesScale = [0, wf.sz*wf.m_per_pix, 0, wf.sz*wf.m_per_pix]
		plt.figure()
		plt.imshow(psf, extent=axesScale)
		plt.title('PSF of optical system')
	return psf

""" Wavefronts """
# Pupil
dOut = 0.7			# outer annulus diameter
dIn = 0.3 			# inner annulus diameter
wavefrontPupil = {	'type':'annulus',\
					'dout': dOut,\
					'din' : dIn\
					}
# Pixel scale
sz = 256			# side length
# Sampling at Nyquist: want each pixel = 1/2 * lambda/D
mPerPix = 1.25 * dOut / sz	# pixel scale
axesScale = [0, sz*mPerPix, 0, sz*mPerPix]
# Optical properties
wavelength_sensing = 589e-9		# Sensing wavelength
wavelength_science = 1064e-9	# Imaging wavelength
# Seeing conditions
r0_500nm = 5e-2					# Corresponds to 'good seeing' at 500 nm. Used here to calculate r0 at other wavelengths
r0_sensing = [np.power((wavelength_sensing / 500e-9), 1.2) * r0_500nm]
r0_science = [np.power((wavelength_science / 500e-9), 1.2) * r0_500nm]

wf_sensing = pyxao.Wavefront(
	wave=wavelength_sensing,
	m_per_pix=mPerPix,
	sz=sz,
	pupil=wavefrontPupil
	)

wf_science = pyxao.Wavefront(
	wave=wavelength_science,
	m_per_pix=mPerPix,
	sz=sz,
	pupil=wavefrontPupil
	)

wavefronts = [wf_sensing, wf_science]

# Printing & plotting wavefront data
printDivider()
print 'Wavefront'
printDivider()
print 'Size (pixels)\t\t', wf_sensing.sz
print 'Metres per px\t\t', wf_sensing.m_per_pix
print 'Sensing wavelength (nm)\t', wf_sensing.wave*1e9
print 'Science wavelength (nm)\t', wf_science.wave*1e9

"""
newFigure()
plt.title('Wavefront pupil')
plt.imshow(wf_sensing.pupil,extent=axesScale)
plt.xlabel('x (m)')
plt.ylabel('y (m)')
"""

""" Wavefront sensor """
nLenslets = 10
pxPerSubap = int(np.floor(wf_sensing.sz / nLenslets))
lensletPitch = pxPerSubap * mPerPix
geometry = 'square'

wfs = pyxao.ShackHartmann(
	mask=None,
	geometry=geometry,
	wavefronts=wavefronts,
	central_lenslet=False,
	lenslet_pitch=lensletPitch,
	sampling=1.0,
	plotit=False,
	weights=None
	)

# Printing & plotting WFS data
printDivider()
print 'Wavefront sensor'
printDivider()
print 'Geometry\t\t', geometry
print 'Lenslets (total)\t', wfs.nlenslets
print 'Lenslet pitch\t\t', wfs.lenslet_pitch
print 'Pixels per lenslet\t', wfs.lenslet_pitch/wavefronts[0].m_per_pix
print 'Focal length\t\t', wfs.flength
print 'Total # of measurements\t', wfs.nsense

""" Deformable mirror """
nActuators = 8
influenceFunction = 'gaussian'
actuatorPitch = wf_sensing.sz / nActuators * mPerPix
dm  = pyxao.DeformableMirror(
	influence_function=influenceFunction,
	wavefronts=wavefronts,
	central_actuator=False,
	plotit=False,
	actuator_pitch=actuatorPitch,
	geometry=geometry,
	edge_radius=1.4
	)

# Printing & plotting DM data
printDivider()
print 'Deformable Mirror'
printDivider()
print 'Geometry\t\t', geometry
print 'Influence function\t', dm.influence_function
print 'Actuators (total)\t', dm.nactuators 
print 'Actuator pitch (m)\t', dm.actuator_pitch

"""
newFigure()
plt.plot(wfs.px[:,0]*wf_sensing.m_per_pix, wfs.px[:,1]*wf_sensing.m_per_pix, 'ro', label="WFS lenslets")
plt.hold(True)
plt.plot(dm.px[:,0]*wf_sensing.m_per_pix, dm.px[:,1]*wf_sensing.m_per_pix, 'yo', label="DM actuators")
plt.xlabel('x (m)')
plt.ylabel('y (m)')
plt.legend()
plt.title('SH WFS and DM geometry')
"""

""" AO system """
ao = pyxao.SCFeedBackAO(
	dm=dm,
	wfs=wfs,
	image_ixs=[1]
	)

pokeStroke = 1e-7	
ao.find_response_matrix(
	mode='onebyone',
	amplitude=pokeStroke
	)
ao.compute_reconstructor( 
	mode='eigenclamp',
	threshold=0.2
	)

"""
newFigure()
plt.subplot(121)
plt.imshow(ao.response_matrix)
plt.xlabel('Actuator number')
plt.ylabel('WFS measurements')
plt.title('AO system response matrix')
plt.colorbar()
plt.subplot(122)
plt.imshow(ao.reconstructor)
plt.ylabel('Actuator number')
plt.xlabel('WFS measurements')
plt.title('AO system response matrix')
plt.colorbar()
"""

# When to add atmosphere?
# Add after calculating the response matrix?

""" Atmosphere """
# Atmospheric conditions 
elevations = [5000.0]	# turbulent layer elevations
angleWind = [np.deg2rad(45.0)]		# angle of the wind (w.r.t. x-axis)
airmass = [1.0]		# air mass
vWind = [0.5]		# wind speed     

printDivider()
print 'Atmosphere'
printDivider()
print 'Elevation (m)\t\t', elevations[0]
print 'Wind speed (m/s)\t', vWind[0]
print 'Wind angle (deg)\t', np.rad2deg(angleWind[0])
print 'Airmass\t\t\t',	airmass[0]
print 'r0 (science) (m)\t', r0_science[0]
print 'r0 (sensing) (m)\t', r0_sensing[0]

atm_sensing = pyxao.Atmosphere(
	sz=sz,
	m_per_pix=mPerPix,
	elevations=elevations,
	v_wind=vWind,
	r_0=r0_sensing,
	angle_wind=angleWind,
	airmass=airmass
	)

atm_science = pyxao.Atmosphere(
	sz=sz,
	m_per_pix=mPerPix,
	elevations=elevations,
	v_wind=vWind,
	r_0=r0_science,
	angle_wind=angleWind,
	airmass=airmass
	)

""" Calculate the PSF of the telescope """
psf = findPSF(wf_sensing, plotit=True)
psf_peak = np.max(psf)

# Adding the atmospheres to the wavefronts
wf_sensing.add_atmosphere(atm_sensing)
wf_science.add_atmosphere(atm_science)

# ao.correct_twice(plotit=True)
ao.run_loop(
	dt=0.002,
	nphot=None,
	niter=500,
	nframesbetweenplots=10,
	plotit=True,
	gain=1.0,
	dodgy_damping=0.9
	)

"""
TODO:
	- Read LQG paper
	- Review relevant AMME5520 material
	- Implement metrics for AO system performance:
		- Strehl ratio
		- Frequency response 
	- what system settings should this be tested under? e.g. aperture size, lenslets, etc.
	- how does wavefront sensing work at different wavelengths?	
		- the WFS detector measures at all wavelengths specified 
		in the vector of wavefronts. 
		That is, the detector image is the superposition of each image
		generated by each wavefront regardless of wavelength.
		This ought to be fixed - allow specification of which wavelengths 
		are used in sensing instead of using all of them to make it
		more realistic.
		- the DM correction applies the same 'distance' correction to
		each wavelength (code seems correct).
	Strehl ratio:
	- need to find the peak intensity in the PSF
	- need to find the peak intensity in the corrected science image
"""