from __future__ import division, print_function
import opticstools as ot
import numpy as np
import matplotlib.pyplot as plt
import ipdb
import scipy.ndimage as nd

# Base WFS class.
# 'ideal & perfect' WFS which simply returns the phase of the wavefront.
# A mask can be used if desired.
class WFS():
    """This is a base wavefront sensor class. 
    
    It is the most abstract perfect wavefront sensor, that returns the phase of the wavefront possibly within a mask. """

    def __init__(self,wavefronts=[],mask=None):
        self.wavefronts = wavefronts  # AZ
        self.waves=[w.wave for w in wavefronts]
        self.mask=mask
        self.pix_to_use=np.where(mask)

    def sense(self, idx=0):
        """ Simply returns the phase of the idx-th wavefront in the list of wavefronts.

        Parameters
        ----------
        idx: int
            Index of the wavefront in the wavefront sensor's array of wavefronts correspoding to the returned phase.
        """
        if self.mask != None:   
            phase = np.angle(self.wavefronts[idx].field)*self.waves[idx]/2/np.pi
            return phase * self.mask
        else:
            return phase

class ShackHartmann(WFS):
    """A Shack-Hartmann wavefront sensor
            
    All lenslets that have their center within the pupil are included.
    
    Parameters
    ----------
    weights: float array
        optional weighting of wavelengths.
    fratio: float
        optional f ratio of the lenslet array.
    N_os: float
        optional Nyquist N_os of the WFS detector. A value of 1 implies Nyquist N_os; a value of 2 implies oversampling by a factor of 2; etc.
    """
    def __init__(
        self,
        plate_scale_as_px,
        nlenslets,
        detector_sz,
        geometry='square',
        wavefronts = [],
        plotit=False,
        weights=None,
        RN = 0,       # read noise
        N_phot = 0    # total # of photons in the WFS detector image
        ):
        
        if len(wavefronts)==0:
            print("ERROR: Must initialise the ShackHartmann with a wavefront list")
            raise UserWarning
        
        self.wavefronts = wavefronts
        self.N_phot = N_phot
        self.RN = RN

        # WFS geometry...
        self.nlenslets = nlenslets
        self.plate_scale_as_px = plate_scale_as_px
        self.detector_sz = detector_sz
        self.geometry = geometry

        if nlenslets % 2 == 1:
            self.central_lenslet = True
        else:
            self.central_lenslet = False

        # Determine the focal length.
        l_px_m = self.wavefronts[0].D / detector_sz
        plate_scale_as_mm = plate_scale_as_px / l_px_m
        self.flength_m = 206265 / plate_scale_as_mm
        lenslet_pitch_px = detector_sz / nlenslets
        lenslet_pitch_m = lenslet_pitch_px * l_px_m

        # Create lenslet geometry
        xpx = []
        ypx = []
        # Hexagonal geometry
        if geometry == 'hexagonal':
             nrows = np.int(np.floor(nlenslets / np.sqrt(3)/2))*2+1 #always odd
             xpx = np.tile(wavefronts[0].sz//2 + (np.arange(nlenslets) - nlenslets//2)*lenslet_pitch_m,nrows)
             xpx = np.append(xpx,np.tile(wavefronts[0].sz//2 - lenslet_pitch_m/2 + (np.arange(nlenslets) - nlenslets//2)*lenslet_pitch_m,nrows-1))
             ypx = np.repeat( wavefronts[0].sz//2 + (np.arange(nrows) - nrows//2)*np.sqrt(3)*lenslet_pitch_m,nlenslets)
             ypx = np.append(ypx,np.repeat( wavefronts[0].sz//2 -np.sqrt(3)/2*lenslet_pitch_m + (np.arange(nrows-1) - nrows//2+1)*np.sqrt(3)*lenslet_pitch_m,nlenslets))
             if not central_lenslet:
                ypx += lenslet_pitch_m/np.sqrt(3)

        # Square geometry        
        elif geometry == 'square':
            # xpx and ypx give the coordinates of each lenslet
            xpx = np.tile(   wavefronts[0].sz//2 + (np.arange(nlenslets) - nlenslets//2)*lenslet_pitch_m,nlenslets)
            ypx = np.repeat( wavefronts[0].sz//2 + (np.arange(nlenslets) - nlenslets//2)*lenslet_pitch_m,nlenslets)
            if not self.central_lenslet:
                xpx += lenslet_pitch_m/2
                ypx += lenslet_pitch_m/2
        else:
            print("ERROR: invalid wavefront sensor geometry")
            raise UserWarning

        #Find pixel values in pupil only.
        px = np.array( [xpx,ypx]).T
        good = np.array([wavefronts[0].pupil[int(np.round(p[1])),int(np.round(p[0]))] != 0 for p in px])
        px = px[good]
        self.px=px
        if plotit:
            #plt.clf()
            plt.figure()
            plt.plot(px[:,0], px[:,1],'o')
        
            # Plot the pupil.
            plt.figure()
            plt.title('WFS pupil')
            plt.imshow(wavefronts[0].pupil)
            plt.show()
                
        #Now go through the wavefronts (i.e. wavelengths) and create the pupil functions
        #and propagators. We first have to find the shortest wavelength (longest focal length)
        if not weights:
            self.weights = np.ones(len(self.wavefronts))
        else:
            self.weights = weights
        self.waves = []
        self.propagator_ixs = []
        # Total number of measurements taken from the wFS
        self.nsense = 3*self.nlenslets #x, y and flux

        self.pupils=[]
        for wf in wavefronts:
            self.waves.append(wf.wave)
            # TODO: I think this is a bug?
            self.propagator_ixs.append(len(wf.propagators))
            # Append a propagator with the appropriate distance.
            wf.add_propagator(self.flength_m)
            # Now the tricky bit... make lots of hexagons!
            pupil = np.zeros( (wf.sz,wf.sz),dtype=np.complex)
            # Make a curved wavefront corresponding to the wavefront over each subaperture.
            # Note that the pupil is actually a complex function because it's a lens - it modifies the wavefront!
            one_lenslet = ot.curved_wf(wf.sz,wf.m_per_px,f_length=self.flength_m,wave=wf.wave)
            # Masking each wavefront to the shape of the subapertures.
            if geometry == 'hexagonal':
                one_lenslet *= ot.utils.hexagon(wf.sz,lenslet_pitch_m)
            elif geometry == 'square':
                one_lenslet *= ot.utils.square(wf.sz,lenslet_pitch_m)
            else:
                print("ERROR: invalid wavefront sensor geometry")
                raise UserWarning

            # Duplicating the wavefronts, one over each lenslet.
            ipdb.set_trace()
            for i in range(self.nlenslets):
                #Shift doens't work on complex arrays, so we have to split into real and
                #imaginary parts.
                to_add = nd.interpolation.shift(one_lenslet.real,(px[i,1]-wf.sz//2,px[i,0]-wf.sz//2),order=1) + \
                    1j * nd.interpolation.shift(one_lenslet.imag,(px[i,1]-wf.sz//2,px[i,0]-wf.sz//2),order=1)
                pupil += to_add
            #The interpolation above decreases amplitude: this restores the amplitude
            nonzero = np.abs(pupil) > 0
            pupil[nonzero] /= np.abs(pupil)[nonzero]
            self.pupils.append(pupil)

        # Create a perfect set of WFS outputs with no atmosphere.
        for wf in wavefronts:
            wf.flatten_field()

        # Get the wavefront sensed by the WFS. At this point the wavefronts have been reset and so all spots lie at the centre of their subapertures. This variable is used in sense() if the desired centroid output format to be specified w.r.t. their nominal positions.
        self.sense_perfect = self.sense(subtract_perfect=False)
            
    def sense(self,mode='gauss_weighted',
            window_hw=5, 
            window_fwhm=5.0, 
            dclamp=10,
            subtract_perfect=True,
            restore_field=True):
        """Sense the tilt and flux modes.

        TODO: modify this so that it doesn't change the field variable in the wavefront instances...
        
        Parameters
        ----------
        N_phot: float
        RN: float
        mode: string
            NOT IMPLEMENTED YET
        window_hw: int
            Half-width of the window to extract the tilt and flux.
        window_fwhm: float
            Full-width half-maximum of the weighting window used to extract each
            lenslet
        RN: float
            Readout noise in electrons.
        dclamp: float
            Denominator clamping value. Gain is reduced for subaperture fluxes below this
            value.
        subtract_perfect: boolean
            Do we subtract the centroid offsets from a perfect (i.e. flat) wavefront?
        restore_field (optional): boolean
            Restore the field after sensing the wavefront, in case someone wants to 
            make an image at this wavelength without re-propagating. i.e. this enables
            a beasmplitter to be placed before the WFS by default. default:True
            
        Returns
        -------
        sensors: (3,nlenslets) numpy array
            An array of sensor positions and flux 
            in pixel or normalised logarithmic flux units.
        """
        sz = self.wavefronts[0].sz
        self.im = np.zeros((sz,sz))

        # Compute the image appearing on the WFS detector.
        for i in range(len(self.wavefronts)):
            # Make a temporary copy of the field so that we can restore it after propgation.
            if restore_field:
                original_field = self.wavefronts[i].field
            
            # Multiply the field by the pupil mask.
            self.wavefronts[i].field = self.wavefronts[i].field*self.pupils[i]
            # Then propagate (Fresnel propagation?)
            self.wavefronts[i].propagate(self.propagator_ixs[i])
            # Add the image component of that wavelength to the final image.
            # The image is generated from the wavefront, the size of which is given by wave_height_px (and has nothing to do with the number of pixels in the WFS detector!)            
            self.im += self.weights[i]*np.abs(self.wavefronts[i].field)**2
            # if i == 0:
            #     self.im = self.weights[i]*self.wavefronts[i].image(N_OS = self.N_os)   
            # else: 
            #     self.im += self.weights[i]*self.wavefronts[i].image(N_OS = self.N_os)

            # So instead, to get the right number of pixels per lenslet, we need to to replace this line with 
            # self.im += self.weights[i]*self.wavefronts[i].image(N_OS = self.N_OS)
            # Then we also need to define N_OS as a property of the WFS. we can calculate it from the plate scale (which in turn is computed from the fratio and the SCALED UP subaperture diameter (so D_out / N_lenslets) and pixel size (so D_subap / N_pixels_per_subap)), D_subap and the wavelength.

            # Restore the field.
            if restore_field:
                self.wavefronts[i].field = original_field
        
        # If the photon number is set, then we add noise.
        if self.N_phot > 0:
            self.im = self.im/np.sum(self.im)*self.N_phot
            self.im = np.random.poisson(self.im).astype(float)
        if self.RN > 0:
            self.im += np.random.normal(scale=self.RN,size=self.im.shape)
        
        # Now sense the centroids.
        xx = np.arange(2*window_hw + 1) - window_hw
        xy = np.meshgrid(xx,xx)
        # make an uninitialized array:
        #   xyf[0] = x coord
        #   xyf[1] = y coord
        #   xyf[2] = image sum (denominator)
        xyf = np.empty( (3,self.nlenslets) )    

        for i in range(self.nlenslets):
            x_int  = int(np.round(self.px[i][0]))
            x_frac = self.px[i][0] - x_int
            y_int  = int(np.round(self.px[i][1]))
            y_frac = self.px[i][1] - y_int
            
            subim = self.im[y_int-window_hw:y_int+window_hw+1,x_int-window_hw:x_int+window_hw+1]
            
            # Gaussian windowing
            gg = np.exp(-((xy[0] - x_frac)**2 + (xy[1] - y_frac)**2)/2.0/window_fwhm*2.3548)

            # Sum of all pixel intensities in the subaperture (denominator of the centroid calc)
            xyf[2,i] = np.sum(subim*gg)
            if self.N_phot > 0:
                denom = np.maximum(xyf[2,i],dclamp)
            else:
                denom = xyf[2,i]
            
            # x and y coords of the centroids
            xyf[0,i] = np.sum((xy[0]-x_frac)*subim*gg)/denom
            xyf[1,i] = np.sum((xy[1]-y_frac)*subim*gg)/denom
        
        #The flux variable only has any meaning with respect to the mean. 
        #A logarithm should also be taken so that differences to a perfect image
        #are meaningful.
        if self.N_phot > 0:
            denom = np.maximum(np.mean(xyf[2]),dclamp)
        else:
            denom = np.mean(xyf[2])
        # normalise the sum w.r.t. the mean value if no photon noise assumed
        xyf[2] /= denom
        xyf[2] = np.log(xyf[2])
        
        if subtract_perfect:
            xyf -= self.sense_perfect
        
        return xyf

class APNLWFS(WFS):
    """The Asymmetric Pupil Non-Linear Wavefront Sensor"""
    def __init__(self,masks=None,pupil=None,m_pix=None,pscale=None,wave=[1e-6],sz=256,diam=25.448):
        """ Initialization can occur with a range of different options.
    
        Parameters
        ----------
        m_pix: float array
            An array of meters per pixel values. There should be one per wavelength, and
            they should be proportional to wavelength.
        """
        if not m_pix:
            #Default to a diversity mask
            try:
                m_pix = 1.0/( 1.0/wave*pscale*sz )
            except:
                print("ERROR: must set either m_pix or wave (array-like),pscale,sz")
        self.sz=sz
        self.m_pix = m_pix
        if not masks:
            masks = np.empty( (len(wave),2,sz,sz),dtype=np.complex)
            for i in range(len(wave)):
                masks[i,:,:,:] = ot.diversity_mask(self.sz,self.m_per_px[i])
        if not pupil:
            pupil = ot.gmt(sz,m_pix=self.m_pix)
        if len(pupil.shape)==1:
            pupil=pupil.reshape( (1,pup.shape[0],pup.shape[1]) )
        self.pupil=pupil
        self.diam=diam


    def ims_from_pup(imsz=19):
        """Create a set of images from a pupil and a set of pupil-plane masks
        """
        #Allow for a pupil at only one wavelength to be input.
        pup=self.pupil
        masks=self.masks
        nmasks = masks.shape[0]
        sz = masks.shape[1]
        xy = np.meshgrid(np.arange(sz)-sz/2, np.arange(sz)-sz/2)
        sincfn = np.sinc(xy[0]/np.float(sz))*np.sinc(xy[1]/np.float(sz))
        sincfn = np.fft.fftshift(sincfn)[:,0:sz/2+1]
        ims = np.zeros( (nmasks,imsz,imsz) )
        for j in range(pup.shape[0]):
            for i in range(nmasks):
                bigim = np.abs(np.fft.fft2(pup[j,:,:]*masks[i,:,:]))**2
                bigim = np.fft.irfft2(np.fft.rfft2(bigim)*sincfn)
                ims[i,:,:] += ( np.roll(np.roll(bigim,imsz//2,axis=0), imsz//2,axis=1) )[0:imsz,0:imsz]
        return ims/np.sum(ims)

    def make_modes(diam, m_pix, tilt_scale,  rescale=True, make_orthogonal=True):
        """Make a set of pupil-plane modes."""
        sz = self.sz
        wave = self.wave
        nmodes = 9
        phase = np.zeros( (nmodes,sz,sz) )
        pmask = np.zeros( (sz,sz) )
        for i in range(6):
          pistons = np.zeros(6)
          pistons[i]=1
          pup = ot.gmt(sz, diam, pistons=pistons)[0,:,:]
          phase[i,:,:] = np.angle(pup)
        pmask = np.abs(pup)
        wz = np.where(pmask == 0)
        #Rather than making things explicitly piston-free, we want to make each pmask pair
        #orghogonal.
        delta = np.sum(pmask*(phase[0,:,:] + phase[1,:,:]))/np.sum(pmask)

        for i in range(6):
            phase[i,:,:] -= delta #np.sum(phase[i,:,:]*pmask)/np.sum(pmask)
            phase[i,wz[0],wz[1]]=0
        # The pistons are orthogonal. But tilt, focus etc are not, so we need to use
        # Gram-Schmidt orthogonalisation. Lets save the components...
        pure_tilt = np.zeros( (2,nmodes) )
        for i in range(2):
            tilts = np.zeros(2)
            tilts[i] = 1.0*tilt_scale
            wf = np.angle(ot.curved_wf(sz,m_pix, power=0, tilt=tilts, wave=wave))
            wf[wz]=0
            if (make_orthogonal):
                for j in range(6):
                    pure_tilt[i,j] = np.sum(phase[j,:,:]*wf)/np.sum(phase[j,:,:]**2)
                    wf -=  pure_tilt[i,j] * phase[j,:,:]
            if (rescale):
                scale = (np.max(wf)-np.min(wf))
                wf /= scale
                pure_tilt[i,:] /= scale
            pure_tilt[i,6+i]=1
            phase[6+i,:,:] = wf
        wf = np.angle(ot.curved_wf(sz,m_pix, defocus=1.0, wave=wave))
        wf -= np.sum(wf*pmask)/np.sum(pmask)
        wf[wz]=0
        if (make_orthogonal):
            for j in range(6):
                wf -= np.sum(phase[j,:,:]*wf)/np.sum(phase[j,:,:]**2) * phase[j,:,:]
        if (rescale):
            wf /= (np.max(wf)-np.min(wf))
        phase[8,:,:] = wf
    
        return pmask, phase, pure_tilt

    def make_ims(params,pmask, modes, masks,imsz=19, wave=2.2e-6, extra_ab=[]):
        """Create a set of images from a set of aberration parameters.
        """
        pup = pmask.copy() + 0j
        wf = np.zeros(pmask.shape)
        for i,p in enumerate(params):
            wf += p*modes[i,:,:]
        #If there is an extra aberration, we remove the components of it that match
        #already parameterised modes
        if len(extra_ab)>0:
            real_mask = np.sum(np.abs(masks),axis=0)
            for i in range(len(params)):
                extra_ab -= modes[i,:,:] * np.sum(modes[i,:,:]*extra_ab*real_mask)/np.sum(modes[i,:,:]**2*real_mask)
        pup *= np.exp(1j*wf)
        return ims_from_pup(pup,masks,imsz=imsz)

#####################################################################
# def __init__(
#         self,
#         lenslet_pitch_m,
#         geometry,
#         central_lenslet,
#         wavefronts = [],
#         N_os = None,   # TODO implement Nyquist N_os!
#         fratio = None,
#         plotit=False,
#         weights=None,
#         RN = 0,       # read noise
#         N_phot = 0    # total # of photons in the WFS detector image
#         ):
        
#         # What do we need to know?
#         #   - the Nyquist N_os (N_OS) of the wavefront sensor, given by
#         #           N_OS = wavelength / 2 / D_subap / plate_scale_rad

#         if len(wavefronts)==0:
#             print("ERROR: Must initialise the ShackHartmann with a wavefront list")
#             raise UserWarning
        
#         self.lenslet_pitch_m = lenslet_pitch_m
#         self.central_lenslet = central_lenslet
#         self.N_phot = N_phot
#         self.RN = RN
#         self.N_os = N_os
        
#         #Create lenslet geometry
#         xpx = []
#         ypx = []
#         # pixels per subaperture
#         lenslet_pitch_m = lenslet_pitch_m/wavefronts[0].m_per_px
#         nlenslets = int(np.floor(wavefronts[0].sz/lenslet_pitch_m))

#         # Hexagonal geometry
#         if geometry == 'hexagonal':
#              nrows = np.int(np.floor(nlenslets / np.sqrt(3)/2))*2+1 #always odd
#              xpx = np.tile(wavefronts[0].sz//2 + (np.arange(nlenslets) - nlenslets//2)*lenslet_pitch_m,nrows)
#              xpx = np.append(xpx,np.tile(wavefronts[0].sz//2 - lenslet_pitch_m/2 + (np.arange(nlenslets) - nlenslets//2)*lenslet_pitch_m,nrows-1))
#              ypx = np.repeat( wavefronts[0].sz//2 + (np.arange(nrows) - nrows//2)*np.sqrt(3)*lenslet_pitch_m,nlenslets)
#              ypx = np.append(ypx,np.repeat( wavefronts[0].sz//2 -np.sqrt(3)/2*lenslet_pitch_m + (np.arange(nrows-1) - nrows//2+1)*np.sqrt(3)*lenslet_pitch_m,nlenslets))
#              if not central_lenslet:
#                 ypx += lenslet_pitch_m/np.sqrt(3)

#         # Square geometry        
#         elif geometry == 'square':
#             # xpx and ypx give the coordinates of each lenslet
#             xpx = np.tile(   wavefronts[0].sz//2 + (np.arange(nlenslets) - nlenslets//2)*lenslet_pitch_m,nlenslets)
#             ypx = np.repeat( wavefronts[0].sz//2 + (np.arange(nlenslets) - nlenslets//2)*lenslet_pitch_m,nlenslets)
#             if not central_lenslet:
#                 xpx += lenslet_pitch_m/2
#                 ypx += lenslet_pitch_m/2
#         else:
#             print("ERROR: invalid wavefront sensor geometry")
#             raise UserWarning

#         #Find pixel values in pupil only.
#         px = np.array( [xpx,ypx]).T
#         good = np.array([wavefronts[0].pupil[int(np.round(p[1])),int(np.round(p[0]))] != 0 for p in px])
#         px = px[good]
#         self.px=px
#         if plotit:
#             #plt.clf()
#             plt.figure()
#             plt.plot(px[:,0], px[:,1],'o')
        
#             # Plot the pupil.
#             plt.figure()
#             plt.title('WFS pupil')
#             plt.imshow(wavefronts[0].pupil)
#             plt.show()
                
#         #Now go through the wavefronts (i.e. wavelengths) and create the pupil functions
#         #and propagators. We first have to find the shortest wavelength (longest focal length)
#         self.wavefronts = wavefronts
#         if not weights:
#             self.weights = np.ones(len(self.wavefronts))
#         else:
#             self.weights = weights
#         self.waves = []
#         self.propagator_ixs = []
#         self.nlenslets = px.shape[0]
#         # Total number of measurements taken from the wFS
#         self.nsense = 3*self.nlenslets #x, y and flux

#         # If the f ratio of the lenslet array is not specified, then figure it out for a given Nyquist N_os.
#         # The f ratio is set as the input instead of the focal length because we scale everything up to the size of the primary mirror, and so having the user input a focal length may be confusing!
#         if not fratio:
#             if N_os:
#                 flength_trial = []
#                 # Calculate the focal length at each wavelength
#                 for wf in wavefronts:
#                     # Compute focal length for the SH WFS
#                     # D = lenslet pitch (diameter of each lenslet)
#                     # f# = f/D, here we assume that f = 1 so f# = 1/D
#                     # resolution = wave/D
#                     # Each pixel is sampled at Nyquist N_os rate so 1 resolution element = 2 pixels
#                     # therefore 2 resolution elements =  wave / D
#                     # therefore 1/D = wave/D/wave = 2 resolution elements/wave, hence the line below
#                     fratio = 2 * wf.m_per_px / wf.wave * N_os
#                     # now we can recover the focal length from the pitch (diameter) and the f#
#                     flength_trial.append(fratio * lenslet_pitch_m)
#                 # find the maximum focal length corresponding to all wavelengths
#                 flength_m = np.max(flength_trial)
#             else:
#                 print("ERROR: you must either specify the fratio or N_os parameters for the WFS!")
#                 raise UserWarning
#         else:
#             if N_os:
#                 print("WARNING: ignoring N_os parameter as fratio has been specified...")
#             flength_m = fratio * lenslet_pitch_m

#         self.flength_m=flength_m
#         self.pupils=[]
#         for wf in wavefronts:
#             self.waves.append(wf.wave)
#             # TODO: I think this is a bug?
#             self.propagator_ixs.append(len(wf.propagators))
#             # Append a propagator with the appropriate distance.
#             wf.add_propagator(flength_m)
#             # Now the tricky bit... make lots of hexagons!
#             pupil = np.zeros( (wf.sz,wf.sz),dtype=np.complex)
#             # Make a curved wavefront corresponding to the wavefront over each subaperture.
#             # Note that the pupil is actually a complex function because it's a lens - it modifies the wavefront!
#             one_lenslet = ot.curved_wf(wf.sz,wf.m_per_px,f_length=flength_m,wave=wf.wave)
#             # Masking each wavefront to the shape of the subapertures.
#             if geometry == 'hexagonal':
#                 one_lenslet *= ot.utils.hexagon(wf.sz,lenslet_pitch_m)
#             elif geometry == 'square':
#                 one_lenslet *= ot.utils.square(wf.sz,lenslet_pitch_m)
#             else:
#                 print("ERROR: invalid wavefront sensor geometry")
#                 raise UserWarning

#             # Duplicating the wavefronts, one over each lenslet.
#             for i in range(self.nlenslets):
#                 #Shift doens't work on complex arrays, so we have to split into real and
#                 #imaginary parts.
#                 to_add = nd.interpolation.shift(one_lenslet.real,(px[i,1]-wf.sz//2,px[i,0]-wf.sz//2),order=1) + \
#                     1j * nd.interpolation.shift(one_lenslet.imag,(px[i,1]-wf.sz//2,px[i,0]-wf.sz//2),order=1)
#                 pupil += to_add
#             #The interpolation above decreases amplitude: this restores the amplitude
#             nonzero = np.abs(pupil) > 0
#             pupil[nonzero] /= np.abs(pupil)[nonzero]
#             self.pupils.append(pupil)

#         # Create a perfect set of WFS outputs with no atmosphere.
#         for wf in wavefronts:
#             wf.flatten_field()

#         # Get the wavefront sensed by the WFS. At this point the wavefronts have been reset and so all spots lie at the centre of their subapertures. This variable is used in sense() if the desired centroid output format to be specified w.r.t. their nominal positions.
#         self.sense_perfect = self.sense(subtract_perfect=False)
