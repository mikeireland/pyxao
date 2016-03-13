from __future__ import division, print_function
import opticstools as ot
import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage as nd
import pdb
plt.ion()

class DeformableMirror():
    """This is the deformable mirror class. """
    def __init__(self,influence_function='gaussian',wavefronts=[],central_actuator=False,\
        plotit=False,actuator_pitch=0.5,geometry='hexagonal',edge_radius=1.4):
        """ The geometry of this class has a lot in common with the ShackHartmann class
        """
        
        if len(wavefronts)==0:
            print("ERROR: Must initialise the DeformableMirror with a wavefront list")
            raise UserWarning
        
        self.wavefronts = wavefronts
        self.actuator_pitch = actuator_pitch
        self.influence_function=influence_function
        
        #Create actuator geometry
        xpx = []
        ypx = []
        lw = actuator_pitch/wavefronts[0].m_per_pix
        nactuators = int(np.floor(wavefronts[0].sz/lw))
        if geometry == 'hexagonal':
             nrows = np.floor(nactuators / np.sqrt(3) )
             xpx = np.tile(wavefronts[0].sz//2 + (np.arange(nactuators) - nactuators//2)*lw,nrows)
             xpx = np.append(xpx,np.tile(wavefronts[0].sz//2 - lw/2 + (np.arange(nactuators) - nactuators//2)*lw,nrows-1))
             ypx = np.repeat( wavefronts[0].sz//2 + (np.arange(nrows) - nrows//2)*np.sqrt(3)*lw,nactuators)
             ypx = np.append(ypx,np.repeat( wavefronts[0].sz//2 -np.sqrt(3)/2*lw + (np.arange(nrows-1) - nrows//2+1)*np.sqrt(3)*lw,nactuators))
             if not central_actuator:
                xpx += lw/2
                ypx += lw*np.sqrt(3)/4
                
        elif geometry == 'square':
            xpx = np.tile(   wavefronts[0].sz//2 + (np.arange(nactuators) - nactuators//2)*lw,nactuators)
            ypx = np.repeat( wavefronts[0].sz//2 + (np.arange(nactuators) - nactuators//2)*lw,nactuators)
            if not central_actuator:
                xpx += lw/2
                ypx += lw/2
        else:
            print("ERROR: invalid actuator geometry")
            raise UserWarning    
        #Find actuators within half an actuator pitch of the pupil only
        px = np.array( [xpx,ypx]).T
        acirc = ot.utils.circle(wavefronts[0].sz,lw*edge_radius)
        good=[]
        for p in px:
            overlap = np.sum(wavefronts[0].pupil*nd.interpolation.shift(acirc,\
                (p[1]-wavefronts[0].sz//2,p[0]-wavefronts[0].sz//2),order=1))
            good.append(np.sum(overlap)>0)
        good = np.array(good)
        #pdb.set_trace()
        px = px[good]
        self.px=px
        if plotit:
            plt.clf()
            plt.plot(px[:,0], px[:,1],'o')
        self.nactuators = px.shape[0]
        
    def apply(self,actuators):
        """Go through all wavefronts and apply this DM"""
        if len(actuators) != len(self.px):
            print("ERROR: Wrong number of actuator values - should be {0:d}".format(len(self.px)))
            raise UserWarning
        sz = self.wavefronts[0].sz
        xx = np.arange(sz) - sz//2
        xy = np.meshgrid(xx,xx)
        rr = np.sqrt(xy[0]**2 + xy[1]**2)
        aw = self.actuator_pitch/self.wavefronts[0].m_per_pix
        if self.influence_function == 'gaussian':
            gg = np.exp(-rr**2/2.0/(aw/2.3548)**2)
        else:
            print("ERROR: Undefined influence function")
            raise UserWarning
        phasescreen = np.zeros( (sz,sz) )
        for i in range(len(self.px)):
            phasescreen += nd.interpolation.shift(gg,(self.px[i,1]-sz//2,self.px[i,0]-sz//2),order=1)*actuators[i]
        for wf in self.wavefronts:
            wf.field = wf.field*np.exp(2j*np.pi*phasescreen/wf.wave)
        
