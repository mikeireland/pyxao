from __future__ import division, print_function
import opticstools as ot
import numpy as np
import matplotlib.pyplot as plt
import pdb
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