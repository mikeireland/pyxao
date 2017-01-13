from __future__ import division, print_function
import opticstools as ot
import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage as nd
import scipy.misc
import ipdb
import time
plt.ion()

try:
    import pyfftw
    pyfftw.interfaces.cache.enable()
    pyfftw.interfaces.cache.set_keepalive_time(1.0)
    nthreads = 16 
    print("WARNING: in wavefront.py I am using pyfftw with {:d} threads...".format(nthreads))
except:
    nthreads = 0
    print("WARNING: in wavefront.py I could not import pyfftw, so I am using numpy fft routines instead...")


class Wavefront():
    """This is a base wavefront class. 
    
    It not only stores the wavefront, but 
    includes propagation and application of the telescope pupil function. As
    a start at least, there is only 1 wavelength. 
    
    Parameters
    ----------
    wave: float
        wavelength in meters
    m_per_px: float
        meters per pixel
    sz: int
        size of the wavefront
    pupil: dict
        Pupil type and parameters."""

    def __init__(self, wave, m_per_px, sz, pupil):
        """ Constructor. """
        self.wave = wave
        self.m_per_px = m_per_px
        self.sz = sz

        try:
            ptype = pupil['type']
        except:
            print("Pupil is a dictionary and must have a 'type' keyword")
            raise UserWarning
        if ptype == "annulus":
            self.pupil = ot.utils.circle(self.sz, pupil['dout']/self.m_per_px) - ot.utils.circle(self.sz, pupil['din']/self.m_per_px)
            self.D = pupil['dout']
        elif ptype == "square":
            self.pupil = ot.utils.square(self.sz, pupil['dout']/self.m_per_px) - ot.utils.square(self.sz, pupil['din']/self.m_per_px)
            self.D = pupil['dout']
        else:
            print("Invalid pupil 'type' keyword!")

        #Initialize an empty list of propagators
        self.propagators = []
        #Initialize the wavefront field: uniform (1 + 0j) at all points
        self.field = (1+0j)*np.ones( (sz,sz) )
        # Index of the propagator list corresponding to the atmosphere
        # (zero since there's no atmosphere yet)
        self.atmosphere_start = 0
        self._atm=None
        
    def add_propagator(self,distance, demag=1.0):
        """Add a propagator to the list of propagators
        
        Parameters
        ----------
        distance: float
            distance to propagate
        demag: float
            pupil demagnification"""
        self.propagators.append(ot.FresnelPropagator(self.sz,self.m_per_px,distance*demag**2, self.wave))
        
    def propagate(self, index):
        """Propagate the wavefront, modifying the field appropriately.
        
        Parameters
        ----------
        index: int
            index of the propagator, from the list of propagators stored by this
            instance."""
        if len(self.propagators) <= index:
            print("ERRROR: index out of range")
            raise UserWarning
        self.field = self.propagators[index].propagate(self.field)
        
    def add_atmosphere(self,atm):
        """Add atmosphere propagators to this wavefront.
        
        Parameters
        ----------
        atm: Atmosphere instance
            The atmosphere instance for the propagators"""
        #Check atmosphere is compatible
        if (self.m_per_px != atm.m_per_px):
            print("ERROR: Atmosphere and wavefront must have the same scale")
        self.atmosphere_start = len(self.propagators)
        for dz in atm.dz:
            self.propagators.append(ot.FresnelPropagator(self.sz,self.m_per_px,dz, self.wave))

        # !!! Do we really need the following line? i.e. we need the propagators, but do we need
        # the *atmosphere* ?
        self._atm=atm
        
    def remove_atmosphere(self):
        """Remove atmosphere propagators and link to the atmosphere"""
        if not self._atm:
            print("ERROR: No atmosphere to remove!")
            raise UserWarning
        for i in range(self._atm.nlayers):
            self.propagators.pop(self.atmosphere_start)

        self._atm = None

            
    def atm_field(self, 
        edge_smooth = 16):
        """Find the electric field at the telescope pupil, and place it in the field variable. 

        It is assumed that the atmosphere is loaded into the atmosphere_start-th index of the propagators for this wavefront.

        NOTE: Fresnel propagation!
        
        Parameters
        ----------
        edge_smooth: int
            Smooth this many pixels from the edge when propagating through the atmosphere.
            Needed because the wavefront is *not* periodic after being truncated to the size (sz) of this wavefront."""

        atm = self._atm
        if not atm:
            print("ERROR: Intitialise wavefront with add_atmosphere() first!")
            raise UserWarning

        # Check atmosphere is compatible
        if (self.m_per_px != atm.m_per_px):
            print("ERROR: Atmosphere and wavefront must have the same scale")
            raise UserWarning

        # Initialise the field: a uniform electric field with amplitude 1.
        self.field = (1+0j) * np.ones((self.sz,self.sz))

        for i in range(atm.nlayers):
            # Apply the atmospheric phase. NB on a timing test, this is more time-intensive than the Fresnel propagation.
            # tic = time.time()
            self.field *= np.exp(2j*np.pi*atm.delays[i][:self.sz,:self.sz]/self.wave)

            # Smooth the edges
            self.field[:edge_smooth,:] = 1 + (self.field[:edge_smooth,:]-1)* \
                np.repeat(np.arange(edge_smooth)/edge_smooth,self.sz).reshape(edge_smooth,self.sz)
            self.field[-edge_smooth:,:] = 1 + (self.field[-edge_smooth:,:]-1)* \
                np.repeat(np.arange(edge_smooth)[::-1]/edge_smooth,self.sz).reshape(edge_smooth,self.sz)
            self.field[:,:edge_smooth] = 1 + (self.field[:,:edge_smooth]-1)* \
                np.tile(np.arange(edge_smooth)/edge_smooth,self.sz).reshape(self.sz,edge_smooth)
            self.field[:,-edge_smooth:] = 1 + (self.field[:,-edge_smooth:]-1)* \
                np.tile(np.arange(edge_smooth)[::-1]/edge_smooth,self.sz).reshape(self.sz,edge_smooth)
            #toc = time.time()

            #Fresnel propagate to the next location
            self.field = self.propagators[i+self.atmosphere_start].propagate(self.field)
            #print("t1: {0:6.3f}, t2 {1:6.3f}".format(toc-tic, time.time()-toc))                
        self.field *= self.pupil 
        
    def flatten_field(self):
        """ 
            Resets the wavefront field to a flat wavefront masked by the pupil.
        """
        self.field = (1+0j) * np.ones((self.sz,self.sz))
        self.field *= self.pupil

    def image(self,
        N_OS = None,
        plate_scale_as_px = None,
        return_efield = False,
        plotit = False
        ):
        """ Return an image based on the current field. 
        Note that this routine does NOT modify the field variable. 

        Parameters
        ----------
        return_efield: boolean
            Do we return an electric field? If not, return the intensity.
        N_OS: float
            How to sample the image. A sampling factor of 1.0 implies Nyquist 
            sampling at this wavefront's wavelength: i.e. the plate scale in the
            image, f_x = wavelength / 2D (i.e. 2 pixels across the FWHM). A 
            sampling factor of 2.0 implies twice this resolution, etc.
        plate_scale_as_px: float
            The plate scale of the final image. 
            Only neither or one of N_OS and plate_scale_as_px must be specified.

            TODO: resample the pupil if the fftpad < 1.
            When fftpad = 1, this corresponds to 1 pixel across the FWHM. 
            This is independent of N.
            """

        # Padding the image to obtain the appropriate plate scale or sampling 
        # for the given wavelength.
        if N_OS and plate_scale_as_px:
            print("ERROR: Nyquist sampling and plate scale cannot both be specified!")
            raise UserWarning
        if not N_OS and not plate_scale_as_px:
            # By default, we want to Nyquist sample (2 pixels per FWHM)
            fftpad = 2
        elif plate_scale_as_px:
            plate_scale_rad_px = np.deg2rad(plate_scale_as_px / 3600)
            fftpad = self.wave / self.D / plate_scale_rad_px   # padding factor
            N_OS = fftpad / 2
        else:
            if N_OS >= 1:
                fftpad = 2 * N_OS
            else:
                # If N_OS < 1, then we need to generate the image at a higher 
                # resolution and then downsample it. 
                fftpad = 2
    
        # total padded side length 
        N = np.ceil(self.sz * fftpad).astype(np.int)  

        # Making sure the result is centred in the image plane.
        # This code was adapted from similar code in OOMAO 
        # (I don't really understand why it works, but it seems to, so no harm 
        # = no foul).
        if not np.remainder(N, 2) and np.remainder(self.sz, 2):
            # If N is even but dim is odd, then simply add 1 to N to centre 
            # the result.
            N += 1
        N = max(N, self.sz)
        
        # Array to hold the image.
        zpad = np.zeros((N, N), dtype = np.complex)
        zpad[:self.sz,:self.sz] = self.field

        # PHASE SHIFT        
        if not np.remainder(N, 2):
            # If N is even, then we need to add a phase shift.            
            arr = np.arange(self.sz) * (((not np.remainder(self.sz, 2)) - N) / N)
            u, v = np.meshgrid(arr,arr) 
            fftPhasor = np.ones(self.field.shape, dtype = np.complex) * \
                np.exp( -1j * np.pi * (u + v))        
            # Applying the phase shift.
            zpad[:self.sz,:self.sz] *= fftPhasor

        # Centering the field 
        roll = np.int((N - self.sz) // 2)
        zpad = np.roll(np.roll(zpad, -roll, axis = 1), -roll, axis = 0)

        # Performing the FFT.
        if (nthreads==0):
            efield = np.fft.fft2(zpad)
        else:
            tic = time.time()
            efield = pyfftw.interfaces.numpy_fft.fft2(zpad,threads=nthreads)
            # print("dt = %.5f" % (time.time()-tic))

        # Shift the zero-frequency component to the center of the spectrum.
        efield = np.fft.fftshift(efield)
        # Need to shift again if we've applied the phase shift.
        if not np.remainder(N, 2): 
            efield = np.fft.fftshift(efield)

        irr = np.abs(efield)**2 # Irradiance

        # Downsampling if required.
        if N_OS and (fftpad < 1):
            irr = scipy.misc.imresize(irr, N_OS)
            efield = scipy.misc.imresize(efield.real, N_OS) + \
            1j*scipy.misc.imresize(efield.imag, N_OS)

        if plotit:
            plt.figure()
            plt.imshow(irr)
            plt.title('Irradiance')
            plt.show()

        # Return the electric field or the irradiance.
        if return_efield:
            return efield
        else:
            return irr

    def psf_dl(self, 
        N_OS = None,
        plate_scale_as_px = None,
        plotit = False,
        return_efield = False
        ):
        """  Find the PSF of the wavefront at a given wavelength. 

        Parameters
        ----------
        return_efield: boolean
            Do we return an electric field? If not, return the intensity. """

        # Make the field uniform.
        self.flatten_field()

        # The PSF is then simply the FFT of the pupil.
        psf = self.image(N_OS = N_OS, plate_scale_as_px = plate_scale_as_px, 
            return_efield = return_efield)
        psf /= sum(psf.flatten())
        if plotit:
            axesScale = [0, self.sz*self.m_per_px, 0, self.sz*self.m_per_px]
            plt.figure()
            plt.imshow(psf, extent = axesScale)
            plt.title('PSF of optical system')
        return psf
            

            