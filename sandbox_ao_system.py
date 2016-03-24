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

""" Wavefronts """
# Pupil
dOut = 2.0			# outer annulus diameter
dIn = 0.4 			# inner annulus diameter
wavefrontPupil = {	'type':'annulus',\
					'dout': dOut,\
					'din' : dIn\
					}
# Pixel scale
sz = 256			# side length
mPerPix = 1.25 * dOut / sz	# pixel scale
axesScale = [0, sz*mPerPix, 0, sz*mPerPix]
# Optical properties
wavelength_sensing = 589e-9		# Sensing wavelength
wavelength_science = 1064e-9	# Imaging wavelength

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
print 'Lenslets per side\t', wfs.nlenslets
print 'Lenslet pitch\t\t', wfs.lenslet_pitch
print 'Pixels per lenslet\t', wfs.lenslet_pitch/wavefronts[0].m_per_pix
print 'Focal length\t\t', wfs.flength
print 'Total # of measurements\t', wfs.nsense

""" Deformable mirror """
nActuators = 6
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
	image_ixs=1
	)

# Printing & plotting AO system data
printDivider()
print 'AO system'
printDivider()

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
angleWind = [45.0]		# angle of the wind (w.r.t. x-axis)
r0_science = [10e-2]		# Fried parameter
r0_sensing = [7.5e-2]		# Fried parameter
airmass = [1.0]		# air mass
vWind = [5.0]		# wind speed     

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

# Adding the atmospheres to the wavefronts
wf_sensing.add_atmosphere(atm_sensing)
wf_science.add_atmosphere(atm_science)

# ao.correct_twice(plotit=True)
ao.run_loop(
	dt=0.002,
	nphot=None,
	niter=20,
	plotit=True,
	gain=1.0,
	dodgy_damping=0.9
	)
