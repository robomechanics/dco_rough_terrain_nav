import torch
import torch.nn.functional as F
import sys
from wheeledSim.robotStateTransformation import robotStateTransformation


def negLogLikelihood(predMean,predLogVar,groundTruth):
    negDoublelogNum = torch.sum((groundTruth-predMean)/predLogVar.exp()*(groundTruth-predMean),dim=-1)
    doublelogDen = predMean.shape[-1]*0.79817986835 + predLogVar.sum(dim=-1)
    return (negDoublelogNum + doublelogDen)/2.0
def sampleGaus(predMean,predLogVar):
    return predMean + (predLogVar/2.0).exp()*torch.randn_like(predLogVar)

class probLosses():
    def __init__(self,predictionNetwork,mapParams):
        self.predictionNetwork = predictionNetwork
        self.mapParams = mapParams
    def getSingleStepLoss(self,sample,hidden=None):
        states = robotStateTransformation(sample[0],terrainMap=sample[4],terrainMapParams=self.mapParams[0],senseParams=self.mapParams[1])
        stateInput = states.getPredictionInput()
        mapInput = states.getHeightMap(useChannel=True)
        actionInput = sample[2]
        predictionInputs = [stateInput,mapInput,actionInput]
        predictionMean,predictionLogVar,hidden,dropoutVals = self.predictionNetwork(predictionInputs,hidden,1)
        groundTruth = states.getRelativeState(sample[3])
        return negLogLikelihood(predictionMean,predictionLogVar,groundTruth).mean()
    def getMultiStepLoss(self,sample,numSteps,hidden=None,useMSE=False):
        if numSteps == 1:
            return self.getSingleStepLoss(sample,hidden)
        # this function does multistep prediction starting at each time step through the whole trajectory
        B = sample[0].shape[0] # batch size
        L = sample[0].shape[1]+1-numSteps # will make multistep predictions for this many chunks per batch
        # we want the predictions to be (B,L,numSteps,stateDim)
        # Make prediction for first step and record hidden/cell states of lstm
        allHiddens = [torch.tensor([],device=sample[0].device),torch.tensor([],device=sample[0].device)]
        allPredictionMeans = torch.tensor([],device=sample[0].device)
        allPredictionLogVars = torch.tensor([],device=sample[0].device)
        states = robotStateTransformation(sample[0][:,0:L,:],terrainMap=sample[4],terrainMapParams=self.mapParams[0],senseParams=self.mapParams[1])
        stateInputs = states.getPredictionInput()
        mapInputs = states.getHeightMap(useChannel=True)
        for i in range(L):
            actions = sample[2][:,i:i+1,:]
            predictionInputs = [stateInputs[:,i:i+1,:],mapInputs[:,i:i+1,:],actions]
            predictionMean,predictionLogVar,hidden,dropoutVals = self.predictionNetwork(predictionInputs,hidden,1)
            if i == 0:
                allDropoutVals = dropoutVals
            else:
                for i in range(len(dropoutVals)):
                    allDropoutVals[i] = torch.cat((allDropoutVals[i],dropoutVals[i]),dim=1)
            allPredictionMeans = torch.cat((allPredictionMeans,predictionMean),dim=1)
            allPredictionLogVars = torch.cat((allPredictionLogVars,predictionLogVar),dim=1)
            allHiddens[0] = torch.cat((allHiddens[0],hidden[0].unsqueeze(dim=2)),dim=2)
            allHiddens[1] = torch.cat((allHiddens[1],hidden[1].unsqueeze(dim=2)),dim=2)
        # allPredictions (B,L,dim), allHiddens (numlayers*numdirections,B,L,dim)
        allGroundTruths = states.getRelativeState(sample[3][:,0:L,:]).unsqueeze(2) #(B,L,1,dim)
        states.updateState(sampleGaus(allPredictionMeans,allPredictionLogVars))
        allPredictionMeans = allPredictionMeans.unsqueeze(2) #(B,L,1,dim)
        allPredictionLogVars = allPredictionLogVars.unsqueeze(2) #(B,L,1,dim)
        # do rest of multistep prediction starting at each timestep
        allHiddens[0] = allHiddens[0].reshape(allHiddens[0].shape[0],-1,allHiddens[0].shape[-1])
        allHiddens[1] = allHiddens[1].reshape(allHiddens[1].shape[0],-1,allHiddens[1].shape[-1])
        for i in range(len(allDropoutVals)):
            allDropoutVals[i] = allDropoutVals[i].reshape(-1,1,allDropoutVals[i].shape[-1])
        # handle multi step
        for i in range(1,numSteps):
            stateInputs = states.getPredictionInput()
            mapInputs = states.getHeightMap(useChannel=True)
            actions = sample[2][:,i:L+i,:]
            predictionInputs = [stateInputs.reshape(-1,1,stateInputs.shape[-1]),
                                mapInputs.reshape(-1,1,mapInputs.shape[-2],mapInputs.shape[-1]),
                                actions.reshape(-1,1,actions.shape[-1])]
            predictionMean,predictionLogVar,allHiddens,_ = self.predictionNetwork(predictionInputs,allHiddens,allDropoutVals)
            predictionMean = predictionMean.reshape(stateInputs.shape[:-1]+(-1,))
            predictionLogVar = predictionLogVar.reshape(stateInputs.shape[:-1]+(-1,))
            allPredictionMeans = torch.cat((allPredictionMeans,predictionMean.unsqueeze(2)),dim=2)
            allPredictionLogVars = torch.cat((allPredictionLogVars,predictionLogVar.detach().unsqueeze(2)),dim=2)
            allGroundTruths = torch.cat((allGroundTruths,states.getRelativeState(sample[3][:,i:L+i,:]).unsqueeze(2)),dim=2)
            states.updateState(sampleGaus(predictionMean,predictionLogVar))
        # all... shape is now (B,L,numSteps,dim)
        if useMSE:
            return F.mse_loss(allPredictionMeans,allGroundTruths)
        else:
            return negLogLikelihood(allPredictionMeans,allPredictionLogVars,allGroundTruths).mean()
    def getMultiStepLossMixture(self,sample,numSteps,numParticles,hidden=None,dropoutVals = 1):
        if numSteps == 1:
            return self.getSingleStepLoss(sample,hidden)
        # define start of multistep window in trajectory
        msStart = torch.randint(sample[0].shape[1]+1-numSteps,(1,))
        # initialize hidden
        if msStart > 0:
            states = robotStateTransformation(sample[0][:,0:msStart,:],terrainMap=sample[4],terrainMapParams=self.mapParams[0],senseParams=self.mapParams[1])
            stateInputs = states.getPredictionInput()
            mapInputs = states.getHeightMap(useChannel=True)
            actionInputs = sample[2][:,0:msStart,:]
            predictionInputs = [stateInputs,mapInputs,actionInputs]
            predictionMean,predictionLogVar,hidden,dropoutVals = self.predictionNetwork(predictionInputs,hidden,dropoutVals)
            hidden = [hidden[i].repeat_interleave(numParticles,dim=1) for i in range(len(hidden))]
        # define particle state
        particles = sample[0][:,msStart:msStart+1,:].unsqueeze(1).repeat_interleave(numParticles,dim=1)
        particles = robotStateTransformation(particles,terrainMap=sample[4],terrainMapParams=self.mapParams[0],senseParams=self.mapParams[1])
        actions = sample[2][:,msStart:msStart+numSteps,:].unsqueeze(1).repeat_interleave(numParticles,dim=1)
        # for recording
        allPredictionMeans = torch.tensor([],device = sample[0].device)
        allPredictionLogVars = torch.tensor([],device = sample[0].device)
        allGroundTruths = torch.tensor([],device = sample[0].device)
        # perform prediction
        for i in range(numSteps):
            stateInputs = particles.getPredictionInput()
            mapInputs = particles.getHeightMap(useChannel=True)
            actionInputs = actions[:,:,i:i+1,:]
            #actionInputs = sample[2][:,msStart+i:msStart+i+1,:].unsqueeze(1).repeat_interleave(numParticles,dim=1)
            predictionInputs = [stateInputs,mapInputs,actionInputs]
            predictionMean,predictionLogVar,hidden,dropoutVals = self.predictionNetwork(predictionInputs,hidden,dropoutVals)
            allPredictionMeans = torch.cat((allPredictionMeans,predictionMean),dim=2)
            allPredictionLogVars = torch.cat((allPredictionLogVars,predictionLogVar),dim=2)
            groundTruth = particles.getRelativeState(sample[3][:,msStart+i:msStart+i+1,:].unsqueeze(1).repeat_interleave(numParticles,dim=1))
            allGroundTruths = torch.cat((allGroundTruths,groundTruth),dim=2)
            particles.updateState(sampleGaus(predictionMean,predictionLogVar))
        particleLogLikelihood = -negLogLikelihood(allPredictionMeans,allPredictionLogVars,allGroundTruths)
        #particleLogLikelihood = particleLogLikelihood.reshape(-1,numParticles,particleLogLikelihood.shape[-1])
        maxLogLikelihood = particleLogLikelihood.max(dim=1,keepdim=True)[0]
        normalizedParticleLogLikelihood = particleLogLikelihood-maxLogLikelihood
        meanLogLikelihood = torch.exp(normalizedParticleLogLikelihood).mean(dim=1,keepdim=True).log()+maxLogLikelihood
        return -meanLogLikelihood.mean()




if __name__ == '__main__':
    import numpy as np
    from torch.utils.data import DataLoader
    sys.path.append('../wheeledRobotSimPybullet')
    from trajectoryDataset import trajectoryDataset
    from robotStateTransformation import robotStateTransformation
    from dynamicsModelNetwork import probDynModel
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    dataRootDir = 'data/test1/'
    csv_file_name = 'meta.csv'
    modelSaveDir = 'models/test1/'
    trajLength = 16
    [simParams,cliffordParams,terrainMapParams,terrainParams,senseParams] = np.load(dataRootDir+'allSimParams.npy',allow_pickle=True)
    mapParams = [terrainMapParams,senseParams]
    data = trajectoryDataset(csv_file_name,dataRootDir,sampleLength=trajLength,startMidTrajectory=False,staticDataIndices=[4],device=device)
    modelParams = np.load(modelSaveDir+'modelParams.npy',allow_pickle=True)
    dynamicsModel = probDynModel(*modelParams).to(device)
    pLoss = probLosses(dynamicsModel,mapParams)
    data = iter(DataLoader(data,shuffle=True,batch_size=2))
    pLoss.getMultiStepLossIndividual(next(data),3)
