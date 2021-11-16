from dcoUtils.probLosses import sampleGaus
from dcoUtils.pdTracker import pdTracker
import torch
from wheeledSim.robotStateTransformation import robotStateTransformation

"""
This function makes closed loop trajectory predictions.
"""
def closed_loop_pred(state,actions,terrainMap,mapParams,dynamicsModel,numParticles,trackerGains=None):
    # state: current state of robot. should have shape (1,1,State Dim)
    # actions: actions robot may take should have shape (Opt Batch Size, Trajectory Length, Action Dim)

    """ initialize nominal state/trajectory prediction (deterministic, no trajectory tracker)"""
    nomTraj = state.repeat_interleave(actions.shape[0],dim=0)
    nomState = robotStateTransformation(nomTraj,terrainMap=terrainMap,terrainMapParams=mapParams[0],senseParams=mapParams[1])
    nomHidden = None
    nomDropoutVals = -1 # no dropout
    
    """ initialize particle states/trajectory predictions (non-deterministic, with trajectory tracker)"""
    particleTraj = nomTraj.unsqueeze(1).repeat_interleave(numParticles,dim=1)
    particles = robotStateTransformation(particleTraj,terrainMap=terrainMap,terrainMapParams=mapParams[0],senseParams=mapParams[1])
    particleHidden = None
    particleDropoutVals = 2 # dropout on, varied for 1) batches 2) particles

    """ Forward simulation through time and predict nominal/particle trajectories"""
    for t in range(actions.shape[1]):
        """ do nominal trajectory prediction"""
        stateInput = nomState.getPredictionInput()
        sensingInput = nomState.getHeightMap(useChannel=True)
        action = actions[:,t:t+1,:]
        predictionInput = [stateInput,sensingInput,action] # input to network for state transition prediction
        nomPrediction,_,nomHidden,nomDropoutVals = dynamicsModel(predictionInput,nomHidden,dropoutVals=nomDropoutVals)
        
        """ do particle trajectory prediction"""
        if numParticles>0:
            stateInput = particles.getPredictionInput()
            sensingInput = particles.getHeightMap(useChannel=True)
            # actions are feedforward actions + corrective feedback action. Feedback actions are calculated using nominal trajectory as reference
            fbCorrection = pdTracker(particles.currentState,nomState.currentState.unsqueeze(1).repeat_interleave(numParticles,dim=1),gains=trackerGains)
            action = torch.clamp(action.unsqueeze(1).repeat_interleave(numParticles,dim=1)+fbCorrection,-1,1)
    #        action = torch.clamp(action.unsqueeze(1).repeat_interleave(numParticles,dim=1),-1,1)
            predictionInput = [stateInput,sensingInput,action]
            predictionMean,predictionLogVar,particleHidden,particleDropoutVals = dynamicsModel(predictionInput,particleHidden,particleDropoutVals)
            particlePrediction = sampleGaus(predictionMean,predictionLogVar) # stochastic prediction (sample from predicted distribution)
        
        """update nominal and particle states"""
        nomState.updateState(nomPrediction)
        nomTraj = torch.cat((nomTraj,nomState.currentState),dim=1)
        if numParticles>0:
            particles.updateState(particlePrediction)
            particleTraj = torch.cat((particleTraj,particles.currentState),dim=2)
    return nomTraj,particleTraj

