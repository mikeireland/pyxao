from __future__ import division, print_function

import opticstools as ot
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import ipdb
import scipy.ndimage as nd
import scipy.linalg as la
import matplotlib.cm as cm
import matplotlib.cbook
import time

TENTH_AIRY_RING = 10.25

COLORBAR_FRACTION = 0.046
COLORBAR_PAD = 0.04

plt.ion()

def strehl(psf, psf_dl):
    """ Calculate the Strehl ratio of an aberrated input PSF given the diffraction-limited PSF. """
    return np.amax(psf) / np.amax(psf_dl)
    
def centre_crop(psf, psf_sz_cropped):
    """Crop an image about the center"""
    H,W = psf.shape
    return psf[(H - psf_sz_cropped)//2 : (H - psf_sz_cropped)//2 + psf_sz_cropped + np.mod((H - psf_sz_cropped),2),
               (W - psf_sz_cropped)//2 : (W - psf_sz_cropped)//2 + psf_sz_cropped + np.mod((W - psf_sz_cropped),2)] 


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

    def __init__(self, 
        wavelength_ixs,
        wavefronts, 
        atm):     
        # List of wavefronts.   
        self.wavelength_ixs = wavelength_ixs
        self.atm = atm
        self.wavefronts = wavefronts
        for wf in self.wavefronts:
            wf.add_atmosphere(atm)

    #############################################################################################################
    def psf_dl(self, plate_scale_as_px, 
        band = None,
        psf_ix = None,
        crop = True,    # Whether or not to crop the PSF. If set to False, the below two arguments are irrelevant.
        psf_sz_cropped = None,          # Size of the PSF. By default cropped at the 10th Airy ring
        psf_sigma_limit_N_os = TENTH_AIRY_RING,     # Corresponds to 10 Airy rings
        plotit = False
        ):
        """ 
            Compute and return the diffraction-limited PSF at the specified wavelength and plate scale. 

            By default, the returned PSF is cropped at the 10th Airy ring. 
        """

        psf_ix = self._get_wavelength_ix(band, psf_ix)
        print("Generating the diffraction-limited PSF at wavelength {:.2f} nm...".format(self.wavefronts[psf_ix].wave * 1e9))

        # Generating the PSF
        psf = self.wavefronts[psf_ix].psf_dl(plate_scale_as_px = plate_scale_as_px, plotit = plotit, return_efield = False)
        psf_sz = psf.shape[0]

        # If crop == False, we simply return the PSF at the native size generated by the FFT method that is used to compute the PSF.
        if crop:           
            # How many 'sigmas' we want the returned PSF grid to encompass. By default, the PSF is cropped at the 10th Airy ring. 
            if psf_sz_cropped == None:
                N_OS = self.wavefronts[psf_ix].wave / self.wavefronts[psf_ix].D / 2 / np.deg2rad(plate_scale_as_px / 3600)  # Nyquist sampling (HWHM in pixels)
                psf_sz_cropped = np.ceil(min(psf_sz, 4 * N_OS * psf_sigma_limit_N_os))          
            psf = centre_crop(psf, psf_sz_cropped) 

        if plotit:
            plt.imshow(psf, extent = np.array([-psf_sz_cropped/2, psf_sz_cropped/2, -psf_sz_cropped/2, psf_sz_cropped/2]) * plate_scale_as_px)
            plt.title(r'Diffraction-limited PSF at $\lambda = %.1f$ $\mu$m' % (self.wavefronts[psf_ix].wave * 1e9))
            plt.show()

        return psf

    #############################################################################################################
    def run_loop(self, dt, plate_scale_as_px, 
                psf_ix = None,
                niter = 100,
                psf_sz_cropped = None,          # Size of the PSF. By default cropped at the 10th Airy ring
                psf_sigma_limit_N_os = TENTH_AIRY_RING,     # Corresponds to 10 Airy rings
                plot_sz_px = 80,           # This is only used in plotting
                nframesbetweenplots = 1,
                plotit = True
                ):
        """Run an AO servo loop.
        
        It is assumed that all wavefronts share the same atmosphere!
        
        Parameters
        ----------
        dt: float
            Time between samples
        niter: int
            Number of iterations of the loop.
        plotit: boolean
            Do we plot potentially useful outputs? NB, this is all pretty slow with
            matplotlib.
        psf_ix: int
            Index in self.wavelengths corresponding to the wavelength of the PSF to be returned.
        band: string
            Imaging band corresponding to the PSF to be returned. Either psf_ix OR band must be specified.
        plate_scale_as_px: float
            Plate scale for the output science images. For now, only put in the plate scale instead of the Nyquist sampling because we image at a range of wavelengths with the same detector.
        """

        print("Generating PSFs at wavelength {:.2f} nm...".format(self.wavefronts[psf_ix].wave * 1e9))

         # Figuring out the size of the PSF image to return.
        psf_sz = self.wavefronts[psf_ix].image(plate_scale_as_px=plate_scale_as_px).shape[0]
        if plate_scale_as_px and not psf_sz_cropped:
            # Oversampling factor of the PSF. Note that N_OS is equivalent to the 'sigma' of the pupil image in the DL.
            plate_scale_rad_px = np.deg2rad(plate_scale_as_px / 3600)
            N_OS = self.wavefronts[psf_ix].wave / self.wavefronts[psf_ix].D / 2 / plate_scale_rad_px
            psf_sz_cropped = np.ceil(min(psf_sz, 4 * N_OS * psf_sigma_limit_N_os))
        else:
            if psf_sz_cropped:
                psf_sz_cropped = np.ceil(min(psf_sz, psf_sz_cropped))        
            else:
                psf_sz_cropped = psf_sz

        # Arrays to hold images
        psfs_cropped = np.zeros((niter, psf_sz_cropped, psf_sz_cropped))# at the psf_ix wavelength


        """ AO Control loop """
        for k in range(niter):  
            #------------------ EVOLVING THE ATMOSPHERE ------------------#
            print("Iteration %d..." % (k+1))
            self.atm.evolve(dt * k)   # This doesn't affect the field variable.
            for wf in self.wavefronts:   # Evolve the atmosphere & update the wavefront fields to reflect the new atmosphere.                                
                wf.atm_field()          # Now, the wavefront fields include only the atmosphere.   
                            
            #-------------------- SAVING PSF ----------------------#        
            # Compute the PSF at the specified index  
            psf = self.wavefronts[psf_ix].image(plate_scale_as_px = plate_scale_as_px) 
            psf /= sum(psf.flatten())
            psfs_cropped[k] = centre_crop(psf, psf_sz_cropped)
                 
            #------------------ PLOTTING ------------------#
            if plotit & ((k % nframesbetweenplots) == 0):
                if k == 0:
                    axes = []
                    plots = []
                    plt.rc('text', usetex=True)                   
                    # Phase
                    fig = plt.figure(figsize=(10,5))  
                    axes.append(fig.add_subplot(1,2,1)) 
                    axes[-1].title.set_text(r'Corrected phase ($\lambda$ = %d nm)' % (self.wavefronts[psf_ix].wave*1e9))
                    plots.append(axes[-1].imshow(np.angle(self.wavefronts[psf_ix].field)*self.wavefronts[psf_ix].pupil,interpolation='nearest', cmap=cm.gist_rainbow, vmin=-np.pi, vmax=np.pi))
                    plt.colorbar(plots[-1],fraction=COLORBAR_FRACTION, pad=COLORBAR_PAD)
                    # Science image
                    axes.append(fig.add_subplot(1,2,2))  
                    axes[-1].title.set_text(r'Science Image ($\lambda$ = %d nm)' % (self.wavefronts[psf_ix].wave*1e9))
                    plots.append(axes[-1].imshow(centre_crop(psf, plot_sz_px),interpolation='nearest', cmap=cm.gist_heat,extent = np.array([-plot_sz_px/2, plot_sz_px/2, -plot_sz_px/2, plot_sz_px/2]) * plate_scale_as_px))
                    plt.gca().xaxis.set_major_formatter(FormatStrFormatter('%.2f\"'))
                    plt.gca().yaxis.set_major_formatter(FormatStrFormatter('%.2f\"'))
                    plt.colorbar(plots[-1],fraction=COLORBAR_FRACTION, pad=COLORBAR_PAD)
                else:
                    # Update the plots
                    fig.suptitle(r'Seeing-limited phase and PSF')
                    plots[0].set_data(np.angle(self.wavefronts[psf_ix].field)*self.wavefronts[psf_ix].pupil)
                    plots[1].set_data(centre_crop(psf, plot_sz_px))
                plt.draw()
                plt.pause(0.00001)  # Need this to plot on some machines.

        return psfs_cropped

    #############################################################################################################
    def _get_wavelength_ix(self, band, ix):
        if band == None and ix == None:
            print("ERROR: you must specify either an imaging band OR an index in the list of wavefronts corresponding to the wavelength of the PSF you want returned!")
            raise UserWarning
        elif band:
            return self.wavelength_ixs[band]
        else:
            return ix