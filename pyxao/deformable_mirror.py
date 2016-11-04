from __future__ import division, print_function
import opticstools as ot
import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage as nd
try:
    import ipdb
except:
    import pdb

class DeformableMirror():
    """A deformable mirror.
 
    The geometry of this class has a lot in common with the ShackHartmann class

    Parameters
    ----------
    influence_function: string
        Type of influence function of each actuator.
    wavefronts: list of Wavefront instances
    central_actuator: boolean
        Is there an actuator at the pupil center? If not, offset by half an actuator.
    plotit: boolean
        Do we make pretty plots?
    actuator_pitch: float
        Separation between actuators in m
    geometry: string
        Actuator geometry. 'hexagonal' or 'square'
    edge_radius: float
        Do we count actuators beyond the edge? Yes if they are within 
        edge_radius * (actuator_pitch/2) of an illuminated part of the pupil."""
        
    # Upon construction, the actuator geometry is set. Nothing else happens really.
    def __init__(self,influence_function='gaussian',wavefronts=[],central_actuator=False,\
        plotit=False,actuator_pitch=0.5,geometry='hexagonal',edge_radius=1.4):

        if len(wavefronts)==0:
            print("ERROR: Must initialise the DeformableMirror with a wavefront list")
            raise UserWarning
        
        self.wavefronts = wavefronts
        self.actuator_pitch = actuator_pitch
        self.influence_function=influence_function
        
        #Create actuator geometry
        xpx = []
        ypx = []
        # Pixels per actuator
        lw = actuator_pitch/wavefronts[0].m_per_px
        nactuators = int(np.floor(wavefronts[0].sz/lw))
        if geometry == 'hexagonal':
             nrows = np.int(np.floor(nactuators / np.sqrt(3)/2))*2+1 #Always odd
            # x, y coordinates of each actuator
             xpx = np.tile(wavefronts[0].sz//2 + (np.arange(nactuators) - nactuators//2)*lw,nrows)
             xpx = np.append(xpx,np.tile(wavefronts[0].sz//2 - lw/2 + (np.arange(nactuators) - nactuators//2)*lw,nrows-1))
             ypx = np.repeat( wavefronts[0].sz//2 + (np.arange(nrows) - nrows//2)*np.sqrt(3)*lw,nactuators)
             ypx = np.append(ypx,np.repeat( wavefronts[0].sz//2 -np.sqrt(3)/2*lw + (np.arange(nrows-1) - nrows//2+1)*np.sqrt(3)*lw,nactuators))
             if not central_actuator:
                ypx += lw/np.sqrt(3)
                
        elif geometry == 'square':
            # x, y coordinates of each actuator
            xpx = np.tile(   wavefronts[0].sz//2 + (np.arange(nactuators) - nactuators//2)*lw,nactuators)
            ypx = np.repeat( wavefronts[0].sz//2 + (np.arange(nactuators) - nactuators//2)*lw,nactuators)
            if not central_actuator:
                xpx += lw/2
                ypx += lw/2
        else:
            print("ERROR: invalid actuator geometry")
            raise UserWarning    
        # Find actuators within half an actuator pitch of the pupil only
        # Coordinates of the actuators
        px = np.array( [xpx,ypx]).T
        # Syntax: array dimensions, diameter
        acirc = ot.utils.circle(wavefronts[0].sz,lw*edge_radius)
        good=[]
        for p in px:
            overlap = np.sum(wavefronts[0].pupil*nd.interpolation.shift(\
                    acirc,\
                    (p[1]-wavefronts[0].sz//2,p[0]-wavefronts[0].sz//2),\
                    order=1))
            good.append(np.sum(overlap)>0)
        good = np.array(good)
        px = px[good]
        self.px=px
        if plotit:
            plt.clf()
            plt.plot(px[:,0], px[:,1],'o')
        self.nactuators = px.shape[0]
        
    def apply(self,coefficients):
        """Go through all wavefronts and apply this DM
        
        Parameters
        ----------
        coefficients: array-like
            Actuator positions (coefficients?) in m.
        """
        if len(coefficients) != len(self.px):
            print("ERROR: Wrong number of actuator values - should be {0:d}".format(len(self.px)))
            raise UserWarning
        sz = self.wavefronts[0].sz
        xx = np.arange(sz) - sz//2
        xy = np.meshgrid(xx,xx)
        rr = np.sqrt(xy[0]**2 + xy[1]**2)
        aw = self.actuator_pitch / self.wavefronts[0].m_per_px

        if self.influence_function == 'gaussian':
            gg = np.exp(-rr**2/2.0/(aw/2.3548)**2)
        else:
            print("ERROR: Undefined influence function")
            raise UserWarning
        
        # Phase screen corresponding to the DM surface and wavefront wavelength.
        phasescreen = np.zeros( (sz,sz) )        
        # Construct the phase perturbation induced by the DM surface.
        for i in range(len(self.px)):
            # Adde the contribution from each actuator together.
            # Note: phasescreen is in units of m
            phasescreen += nd.interpolation.shift(gg,(self.px[i,1]-sz//2,self.px[i,0]-sz//2),order=1)*coefficients[i]

        # Add the DM correction to the wavefronts.
        for wf in self.wavefronts:
            # The phasescreen gets converted to a true phase: phase = phasecreen * 2pi / lambda
            # So the same correction in terms of distance gets applied to all wavelengths 
            # which corresponds to different phase corrections based on the wavelength.
            wf.field = wf.field * np.exp(2j*np.pi*phasescreen/wf.wave)

        self.phasescreen = phasescreen
        
