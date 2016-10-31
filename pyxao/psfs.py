import numpy as np
import miscutils as mu
from aosim.pyxao import TENTH_AIRY_RING
from wfs import ShackHartmann
from atmosphere import Atmosphere
from wavefront import Wavefront
from deformable_mirror import DeformableMirror
from ao_system import SCFeedBackAO
from linguinesim.imutils import centreCrop
import pdb

####################################################################################################
def getAtmPsfs(wavelength_science_m, N_frames, psf_as_px, dt, D_out, D_in, 
	r0_ref, v_wind_m, wind_angle_deg, elevations_m, airmass,	# Atmospheric parameters
	mode,
	wave_height_px,								# Grid size in FFT (larger = better)
	detector_size_px = (320,256),				# Only used in plotting.
	psf_sigma_limit_N_os = TENTH_AIRY_RING,		# Size of the returned series of PSFs. By default, the PSFs will extend to the 10th Airy ring.
	plotit = False,
	save = False, fname = "ao_psfs"
	):

	return None

####################################################################################################
def getAoPsfs(ao_system, N_frames, psf_as_px, dt,
	psf_sigma_limit_N_os = TENTH_AIRY_RING,		# Size of the returned series of PSFs
	plot_sz = (320,256),						# Only used in plotting
	mode = 'open loop',
	plotit = True,
	saveIt = False,
	fname = "ao_psfs"):
	
	# Determining the response matrix and reconstructor matrix.
	# if mode != 'open loop':
	# 	print("WARNING: I am not recomputing the response and reconstructor matrices - I am assuming they are already initialised in the AO system instance")
	ao_system.find_response_matrix()
	ao_system.compute_reconstructor(threshold=0.1)

	# Running the AO loop.
	psfs_ao, psf_mean, psf_mean_all, strehls = ao_system.run_loop(
		dt = dt,                    
	    mode = mode,
	    niter = N_frames,
	    psf_ix = 1, # Index in the list of wavefronts of the PSF (stored in the DM instance) you want to be returned
	    plate_scale_as_px = psf_as_px,       # Plate scale of the output images/PSFs
	    psf_sigma_limit_N_os = psf_sigma_limit_N_os,		# Extent of the returned PSFs.
	    nframesbetweenplots = 10,
	    plotit = plotit
	    )

	# No turbulence.
	psf_dl = centreCrop(ao_system.dm.wavefronts[1].psf_dl(plate_scale_as_px = psf_as_px), psfs_ao[0].shape)

	# Saving to file
	if saveIt:
		np.savez(fname, 
			ao_system = ao_system,
			N_frames = N_frames,
			psf_as_px = psf_as_px,
			dt = dt,
			psfs_ao = psfs_ao,
			psf_dl = psf_dl,
			strehls = strehls
			)

	# Output the seeing-limited 
	return psfs_ao, psf_dl

####################################################################################################
# def getAoPsfs(wavelength_science_m, N_frames, psf_as_px, dt, D_out, D_in, 
# 	r0_ref, v_wind_m, wind_angle_deg, elevations_m, airmass,	# Atmospheric parameters
# 	mode,
# 	wave_height_px,								# Grid size in FFT (larger = better)
# 	N_actuators=2, dm_geometry='square', central_actuator=False,	# AOI: actuator in the middle
# 	N_lenslets=3, wavelength_wfs_m=589e-9, wfs_geometry='square', central_lenslet=True, # fratio ~60-70 for AOI
# 	detector_size_px = (320,256),				# Only used in plotting.
# 	psf_sigma_limit_N_os = TENTH_AIRY_RING,		# Size of the returned series of PSFs. By default, the PSFs will extend to the 10th Airy ring.
# 	wavelength_ref_m = 500e-9,					# reference wavelength for the supplied r0
# 	plotit = False,
# 	save = False, fname = "ao_psfs"
# 	):
# 	"""
# 		Returns a time series of PSFs (normalised by default) of a telescope with inner and outer primary mirror diameters D_in and D_out respectively in the presence of atmospheric turbulence. 
 
# 		The diffraction-limited PSF of the system is also returned.
# 	"""
# 	wavefrontPupil = {	
# 		'type':'annulus',
# 		'dout': D_out,
# 		'din' : D_in
# 	}

# 	# Wave parameters
# 	m_per_px = D_out / wave_height_px		# Physical mapping of wave onto primary mirror size

# 	# AO system parameters
# 	actuator_pitch_m = D_out / N_actuators
# 	lenslet_pitch_m = D_out / N_lenslets
# 	edge_radius = 1.4	
# 	influence_fun = 'gaussian'
# 	pokeStroke = 1e-7	

# 	# Seeing conditions
# 	r0_wfs = np.power((wavelength_wfs_m / wavelength_ref_m), 1.2) * r0_ref
# 	r0_science = np.power((wavelength_science_m / wavelength_ref_m), 1.2) * r0_ref

# 	####################################################
# 	# Setting up AO system
	# wf_wfs = Wavefront(wave = wavelength_wfs_m, m_per_px = m_per_px, sz = wave_height_px, pupil = wavefrontPupil)
	# wf_science = Wavefront(wave = wavelength_science_m, m_per_px = m_per_px, sz = wave_height_px, pupil = wavefrontPupil)
	# wavefronts_dm = [wf_wfs, wf_science] 	# Wavefronts corrected by the DM (in a CL AO system, it's all of them!)
	# wavefronts_wfs = [wf_wfs]				# Wacefronts sensed by the WFS
	# psf_ix = 1		# Index in the list of wavefronts passed to the DM instance corresponding to the PSF to return

	# dm = DeformableMirror(
	# 	wavefronts = wavefronts_dm, 
	# 	influence_function = 'gaussian', 
	# 	central_actuator = central_actuator, 
	# 	actuator_pitch = actuator_pitch_m, 
	# 	geometry = dm_geometry, 
	# 	edge_radius = 1.4)

	# wfs = ShackHartmann(
	# 	wavefronts = wavefronts_wfs, 
	# 	lenslet_pitch = lenslet_pitch_m, 
	# 	geometry = wfs_geometry, 
	# 	central_lenslet = central_lenslet, 		
	# 	sampling = 1)
	# 	# fratio = wfs_fratio)

	# ao = SCFeedBackAO(dm = dm, wfs = wfs, image_ixs = psf_ix)
	
	# if mode != 'open_loop':
	# 	ao.find_response_matrix()
	# 	ao.compute_reconstructor(threshold=0.1)

	# # The atmosphere is a PHASE SCREEN: it's the same at all wavelengths! We don't need to make a new atmosphere instance for each wavelength
	# # If you need convincing, see minerva_red.py
	# atm = Atmosphere(sz = wave_height_px, m_per_px = m_per_px,
	# 	elevations = elevations_m, r_0 = r0_ref, wave_ref = 500e-9, angle_wind = wind_angle_deg,
	# 	v_wind = v_wind_m, airmass = airmass, seed = 3)

	# wf_wfs.add_atmosphere(atm)
	# wf_science.add_atmosphere(atm)

	# # Calculating the Nyquist oversampling factor
	# psf_rad_px = np.deg2rad(psf_as_px / 3600)
	# N_OS = wf_science.wave / D_out / 2 / psf_rad_px

	# # Running the AO loop.
	# psfs_cropped, psf_mean, psf_mean_all, strehls = ao.run_loop(dt = dt,                    # WFS photon noise.
 #                mode = mode,
 #                niter = N_frames,
 #                psf_ix = psf_ix,                     # Index in the list of wavefronts of the PSF you want to be returned
 #                plate_scale_as_px = psf_as_px,       # Plate scale of the output images/PSFs
 #                psf_sigma_limit_N_os = psf_sigma_limit_N_os,		# Extent of the returned PSFs.
 #                nframesbetweenplots = 1,
 #                plotit = plotit
 #                )

	# # No turbulence.
	# psf_dl = centreCrop(wf_science.psf_dl(plate_scale_as_px = psf_as_px), psfs_cropped[0].shape)

	# # Saving to file
	# if save:
	# 	np.savez(fname, 
	# 		N_frames = N_frames,
	# 		psf_atm = psfs_cropped,
	# 		psf_dl = psf_dl,
	# 		strehls = strehls,
	# 		plate_scale_as_px = psf_as_px,
	# 		N_OS = N_OS,
	# 		dt = dt,
	# 		wavelength_m = wavelength_science_m, 
	# 		r0_ref = r0_ref,
	# 		r0_science = r0_science,
	# 		r0_wfs = r0_wfs,
	# 		elevations = elevations_m,
	# 		v_wind = v_wind_m,
	# 		wind_angle = wind_angle_deg,
	# 		airmass = airmass,
	# 		wave_height_px = wave_height_px,
	# 		D_out = D_out,
	# 		D_in = D_in
	# 		)

	# # Output the seeing-limited 
	# return psfs_cropped, psf_dl