from __future__ import division, print_function

from linguinesim.obssim import convolvePSF, strehl
from linguinesim.imutils import centreCrop
import opticstools as ot
import numpy as np
import matplotlib.pyplot as plt
import pdb
import scipy.ndimage as nd
import scipy.linalg as la
import matplotlib.cm as cm
import miscutils as mu
import time
from aosim.pyxao import TENTH_AIRY_RING

plt.ion()

class SeeingLimitedOpticalSystem():
    """A single-congugate adaptive optics system. The initialization order should be:
    
    1) Wavefront() for every wavelength.
    2) A wavefront sensor.
    3) A deformable mirror.
    4) This AO system.
    5) An Atmosphere().
    
    By convention, if not set, all wavefronts not used for sensing are
    used to create separate science images.
    
    Parameters
    ----------
    dm: DeformableMirror instance
    wfs: WFS or child class instance
    conjugate_location: float
        location of wfs and DM conjugate
    image_ixs: list
        A list of indexes for which images should be calculated, i.e. the 
        science wavefront indices in the dm wavefronts list.
        """

    def __init__(self, wavefronts):        
        self.wavefronts = wavefronts

    #############################################################################################################
    def psf_dl(self, ix, plate_scale_as_px,
        psf_sigma_limit_N_os = TENTH_AIRY_RING,     # Corresponds to 10 Airy rings
        plotIt = False
        ):
        """ 
            Compute and return the diffraction-limited PSF at the specified wavelength and plate scale. 

            By default, the returned PSF is cropped at the 10th Airy ring. 
        """

        # Generating the PSF
        psf = self.wavefronts[ix].psf_dl(plate_scale_as_px = plate_scale_as_px, plotIt = plotIt, return_efield = False)
        psf_sz = psf.shape[0]

        # Cropping if necessary
        N_OS = self.wavefronts[ix].wave / self.wavefronts[ix].D / 2 / np.deg2rad(plate_scale_as_px / 3600)  # Nyquist sampling
        psf_sz_cropped = np.ceil(min(psf_sz, 4 * N_OS * psf_sigma_limit_N_os))
        psf = centreCrop(psf, psf_sz_cropped) 

        if plotIt:
            mu.astroimshow(im = psf, plate_scale_as_px = plate_scale_as_px, title = r'Diffraction-limited PSF at $\lambda = %.1f$ $\mu$m' % (self.wavefronts[ix].wave * 1e9))

        return psf

    #############################################################################################################
    def run_loop(self, dt, plate_scale_as_px, psf_ix,
                niter = 100,
                psf_sz_cropped = None,          # Size of the PSF. By default cropped at the 10th Airy ring
                psf_sigma_limit_N_os = TENTH_AIRY_RING,     # Corresponds to 10 Airy rings
                plot_sz_px = (80,80),           # This is only used in plotting
                nframesbetweenplots = 1,
                plotIt = True
                ):
        """Run an AO servo loop.
        
        It is assumed that all wavefronts share the same atmosphere!
        
        Parameters
        ----------
        dt: float
            Time between samples
        niter: int
            Number of iterations of the loop.
        plotIt: boolean
            Do we plot potentially useful outputs? NB, this is all pretty slow with
            matplotlib.
        psf_ix: int
            Index in self.image_ixs corresponding to the wavelength of the PSF to be returned.
        plate_scale_as_px: float
            Plate scale for the output science images. For now, only put in the plate scale instead of the Nyquist sampling because we image at a range of wavelengths with the same detector.
        """

        # Calculate the DL PSF at the given wavelength.
        psf_dl = self.psf_dl(psf_ix, plate_scale_as_px)

          # Size of the uncropped PSF image.
        psf_sz = self.wavefronts[psf_ix].image(plate_scale_as_px=plate_scale_as_px).shape[0]

        # Oversampling factor of the PSF. Note that N_OS is equivalent to the 'sigma' of the pupil image in the DL.
        N_OS = self.wavefronts[psf_ix].wave / self.wavefronts[psf_ix].D / 2 / np.deg2rad(plate_scale_as_px / 3600)

        # How many 'sigmas' we want the returned PSF grid to encompass.
        if not psf_sz_cropped:
            psf_sz_cropped = np.ceil(min(psf_sz, 4 * N_OS * psf_sigma_limit_N_os))
        else:
            psf_sz_cropped = np.ceil(min(psf_sz, psf_sz_cropped))

        # Arrays to hold images
        psfs_cropped = np.zeros((niter, psf_sz_cropped, psf_sz_cropped))# at the psf_ix wavelength
        psf_mean = np.zeros((psf_sz_cropped, psf_sz_cropped))           # at the psf_ix wavelength


        """ AO Control loop """
        for k in range(niter):  
            #------------------ EVOLVING THE ATMOSPHERE ------------------#
            print("Iteration %d..." % (k+1))
            for wf in self.wavefronts:   # Evolve the atmosphere & update the wavefront fields to reflect the new atmosphere.                
                wf.atm.evolve(dt * k)   # This doesn't affect the field variable.
                wf.atm_field()          # Now, the wavefront fields include only the atmosphere.   
                            
            #-------------------- SAVING PSFS AND SCIENCE IMAGE ----------------------#        
            # Compute the PSF at the specified index  
            psf = self.wavefronts[psf_ix].image(plate_scale_as_px = plate_scale_as_px) 
            
            # Normalising
            psf /= sum(psf.flatten())   

            # Cropping to the specified extent
            psfs_cropped[k] = centreCrop(psf, psf_sz_cropped) 
            
            # Update the mean PSF.
            if (k != 0):
                psf_mean += psfs_cropped[k]

            # Calculate the Strehl ratio.
            sr = strehl(psf = psf, psf_dl = psf_dl)
                 
            #------------------ PLOTTING ------------------#
            if plotIt & ((k % nframesbetweenplots) == 0):                  
                if k == 0:
                    axes = []
                    plots = []
                    fig_width = 2
                    plt.rc('text', usetex=True)
                    fig = mu.newfigure(width=fig_width,height=1)
                    # Phase
                    phase_current = np.angle(self.wavefronts[psf_ix].field) * self.wavefronts[psf_ix].pupil
                    axes.append(fig.add_subplot(1,fig_width,1)) 
                    axes[-1].title.set_text(r'Corrected phase ($\lambda$ = %d nm)' % (self.wavefronts[psf_ix].wave*1e9))
                    plots.append(axes[-1].imshow(phase_current,interpolation='nearest', cmap=cm.gist_rainbow, vmin=-np.pi, vmax=np.pi))
                    mu.colorbar(plots[-1])
                    # PSF
                    axes.append(fig.add_subplot(1,fig_width,2))  
                    axes[-1].title.set_text(r'Point spread function ($\lambda$ = %d nm), Strehl = %.5f' % (self.wavefronts[psf_ix].wave*1e9, sr))
                    plots.append(axes[-1].imshow(centreCrop(psf, plot_sz_px),interpolation='nearest', cmap=cm.gist_heat))
                    mu.colorbar(plots[-1])
                else:
                    fig.suptitle(r'Iteration $k = %d$' % k)    
                    phase_current = np.angle(self.wavefronts[psf_ix].field)*self.wavefronts[psf_ix].pupil
                    plots[0].set_data(phase_current)    # Update phase
                    plots[1].set_data(centreCrop(psf, plot_sz_px))
                    axes[1].title.set_text(r'Point spread function ($\lambda$ = %d nm), Strehl = %.5f' % (self.wavefronts[psf_ix].wave*1e9, sr))

                plt.draw()
                plt.pause(0.00001)  # Need this to plot on some machines.
            #----------------------------------------------#

        # Compute the mean images.
        psf_mean /= niter

        if plotIt:
            mu.newfigure(1,2)
            plt.suptitle('Seeing-limited PSFs')
            mu.astroimshow(centreCrop(psf_dl, psf_sz_cropped), 'Diffraction-limited PSF', plate_scale_as_px, 121)
            mu.astroimshow(psf_mean, r'Mean PSF ($\lambda = %d$ nm), Strehl = %.5f' % (self.wavefronts[psf_ix].wave*1e9, strehl(psf = psf_mean, psf_dl = psf_dl)), plate_scale_as_px, 122)
            plt.show()              

        return psfs_cropped, psf_mean