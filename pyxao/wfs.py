from __future__ import division, print_function
import opticstools as ot
import numpy as np
import matplotlib.pyplot as plt
import pdb
import scipy.ndimage as nd
plt.ion()

class WFS():
    """This is a base wavefront sensor class. It is the most abstract perfect wavefront
    sensor, that returns the phase of the wavefront possibly within a mask. """
    def __init__(self,waves=[1e-6],mask=None):
        self.waves=waves
        self.mask=mask
        self.pix_to_use=np.where(mask)
    def sense(self,wavefront):
        if self.mask():
            return np.angle(wavefront[pix_to_use])*self.waves[0]/2/np.pi
        else:
            return np.angle(wavefront)*self.waves[0]/2/np.pi

class ShackHartmann(WFS):
    """A Shack-Hartmann wavefront sensor"""
    def __init__(self,mask=None,geometry='hexagonal',wavefronts=[],central_lenslet=True,
        lenslet_pitch=0.5,sampling=1.0,plotit=False,weights=None):
        """Initialise the wavefront sensor.
        
        All lenslets that have their center within the pupil are included.
        
        Parameters
        ----------
        sampling: float
            wavefront sensor sampling as a multiple of nyquist
        weights: float array
            optional weighting of wavelengths.
        """
        if len(wavefronts)==0:
            print("ERROR: Must initialise the ShackHartmann with a wavefront list")
            raise UserWarning
        
        self.lenslet_pitch=lenslet_pitch
        
        #Create lenslet geometry
        xpx = []
        ypx = []
        lw = lenslet_pitch/wavefronts[0].m_per_pix
        nlenslets = int(np.floor(wavefronts[0].sz/lw))
        if geometry == 'hexagonal':
             nrows = np.floor(nlenslets / np.sqrt(3) )
             xpx = np.tile(wavefronts[0].sz//2 + (np.arange(nlenslets) - nlenslets//2)*lw,nrows)
             xpx = np.append(xpx,np.tile(wavefronts[0].sz//2 - lw/2 + (np.arange(nlenslets) - nlenslets//2)*lw,nrows-1))
             ypx = np.repeat( wavefronts[0].sz//2 + (np.arange(nrows) - nrows//2)*np.sqrt(3)*lw,nlenslets)
             ypx = np.append(ypx,np.repeat( wavefronts[0].sz//2 -np.sqrt(3)/2*lw + (np.arange(nrows-1) - nrows//2+1)*np.sqrt(3)*lw,nlenslets))
             if not central_lenslet:
                xpx += lw/2
                ypx += lw*np.sqrt(3)/4
                
        elif geometry == 'square':
            xpx = np.tile(   wavefronts[0].sz//2 + (np.arange(nlenslets) - nlenslets//2)*lw,nlenslets)
            ypx = np.repeat( wavefronts[0].sz//2 + (np.arange(nlenslets) - nlenslets//2)*lw,nlenslets)
            if not central_lenslet:
                xpx += lw/2
                ypx += lw/2
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
            plt.plot(px[:,0], px[:,1],'o')
        
        #Now go through the wavefronts (i.e. wavelengths) and create the pupil functions
        #and propagators. We first have to find the shortest wavelength (longest focal length)
        self.wavefronts = wavefronts
        if not weights:
            self.weights = np.ones(len(self.wavefronts))
        else:
            self.weights = weights
        self.waves = []
        self.propagator_ixs = []
        self.nlenslets = px.shape[0]
        flength_trial = []
        for wf in wavefronts:
            #Compute focal length.
            fratio = 2*wf.m_per_pix/wf.wave
            flength_trial.append(fratio*lenslet_pitch)
        flength = np.max(flength_trial)
        self.flength=flength
        self.pupils=[]
        for wf in wavefronts:
            self.waves.append(wf.wave)
            self.propagator_ixs.append(len(wf.propagators))
            wf.add_propagator(flength)
            #Now the tricky bit... make lots of hexagons!
            pupil = np.zeros( (wf.sz,wf.sz),dtype=np.complex)
            one_lenslet = ot.curved_wf(wf.sz,wf.m_per_pix,f_length=flength,wave=wf.wave)
            if geometry == 'hexagonal':
                one_lenslet *= ot.utils.hexagon(wf.sz,lw)
            elif geometry == 'square':
                one_lenslet *= ot.utils.hexagon(wf.sz,lw)
            else:
                print("ERROR: invalid wavefront sensor geometry")
                raise UserWarning
            
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
            
    def sense(self,mode='gauss_weighted',window_hw=5, window_fwhm=7.0):
        """Sense the tilt and flux modes.
        
        Parameters
        ----------
        mode: string
            NOT IMPLEMENTED YET
        """
        sz = self.wavefronts[0].sz
        self.im = np.zeros( (sz,sz) )
        for i in range(len(self.wavefronts)):
            self.wavefronts[i].field = self.wavefronts[i].field*self.pupils[i]
            self.wavefronts[i].propagate(self.propagator_ixs[i])
            self.im += self.weights[i] + np.abs(self.wavefronts[i].field)**2
        
        #Now sense the centroids.
        xx = np.arange(2*window_hw + 1) - window_hw
        xy = np.meshgrid(xx,xx)
        xyf = np.empty( (3,self.nlenslets) )
        for i in range(self.nlenslets):
            x_int  = int(np.round(self.px[i][0]))
            x_frac = self.px[i][0] - x_int
            y_int  = int(np.round(self.px[i][1]))
            y_frac = self.px[i][1] - y_int
            subim = self.im[y_int-window_hw:y_int+window_hw+1,x_int-window_hw:x_int+window_hw+1]
            gg = np.exp(-((xy[0] - x_frac)**2 + (xy[1] - y_frac)**2)/2.0/window_fwhm*2.3548)
            xyf[2,i] = np.sum(subim*gg)
            xyf[0,i] = np.sum((xy[0]-x_frac)*subim*gg)/xyf[2,i]
            xyf[1,i] = np.sum((xy[1]-y_frac)*subim*gg)/xyf[2,i]
        
        #The flux variable only has any meaning with respect to the mean. 
        #A logarithm should also be taken so that differences to a perfect image
        #are meaningful.
        xyf[2] /= np.mean(xyf[2])
        xyf[2] = np.log(xyf[2])
        
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
                masks[i,:,:,:] = ot.diversity_mask(self.sz,self.m_per_pix[i])
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