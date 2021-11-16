import os
import argparse
import yaml
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from dcoUtils.dynamicsModelNetwork import probDynModel
from dcoUtils.probLosses import probLosses
from wheeledSim.trajectoryDataset import trajectoryDataset
from wheeledSim.robotStateTransformation import robotStateTransformation

class sampleLoader(object):
    """ This function loads data for training"""
    def __init__ (self,data,batchSize):
        self.data = data
        self.batchSize = batchSize
        self.samples = iter([])
    def getSample(self):
        try:
            sample = next(self.samples)
        except StopIteration:
            self.samples = iter(DataLoader(self.data,shuffle=True,batch_size=self.batchSize))
            sample = next(self.samples)
        return sample

if __name__ == '__main__':
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    
    """ load arguments for training"""
    parser = argparse.ArgumentParser(description="Trains probabilistic dynamics model")
    parser.add_argument('-c','--config',help='path to config directory',default='config/')
    parser.add_argument('-d','--data_dir',help='path to training data directory',default='training_data/')
    parser.add_argument('-m','--model_file',help='file to save trained model',default='models/trained_model.pt')
    parser.add_argument('-ss_m','--ss_model_file',help='file to save single step model. By default, not saved.',default='')
    args = parser.parse_args()

    """ load training data """
    trajLength = 64
    dataRootDir = args.data_dir
    csv_file = 'meta.csv'
    data = trajectoryDataset(csv_file,dataRootDir,sampleLength=trajLength,startMidTrajectory=False,staticDataIndices=[4],device=device)
    
    """ load parameters """
    model_params = yaml.safe_load(open(os.path.join(args.config,'model_params.yaml'),'r'))
    sim_params = yaml.safe_load(open(os.path.join(args.config,'sim_params.yaml'),'r'))
    numParticles = int(model_params['trainParams']['numParticles'])
    numSteps = model_params['trainParams']['startingSteps']
    stepIncreaseIters = np.cumsum(model_params['trainParams']['stepIncreaseIters'])
    lr = float(model_params['trainParams']['lr'])
    batchSize = model_params['trainParams']['batchSize']
    mapParams=[sim_params['terrainMapParams'],sim_params['senseParams']]
    
    """ create dynamics model"""
    dynamicsModel = probDynModel(model_params).to(device)
    dynamicsModel.train()
    
    """ for actual training """
    optimizer = Adam(dynamicsModel.parameters(),lr=lr)
    lossFunc = probLosses(dynamicsModel,mapParams)
    writer = SummaryWriter(comment="_numSteps:"+str(len(stepIncreaseIters)+numSteps-1)+"_numParticles:"+str(numParticles))
    trainSampler = sampleLoader(data,batchSize)

    for iteration in range(stepIncreaseIters[-1]):
        if iteration > stepIncreaseIters[numSteps-1]:
            numSteps+=1
        sample = trainSampler.getSample()
        trainLoss = lossFunc.getMultiStepLossMixture(sample,numSteps,numParticles=numParticles)
        optimizer.zero_grad()
        trainLoss.backward()
        optimizer.step()
        if iteration%10==0:
            print("iteration: " +str(iteration) + " num steps: " + str(numSteps))
            writer.add_scalar('train/predictionLoss',trainLoss.item(),iteration)
        if iteration%1000==0:
            torch.save(dynamicsModel.state_dict(),args.model_file)
            if len(args.ss_model_file)>0 and numSteps==1:
                torch.save(dynamicsModel.state_dict(),args.ss_model_file)
            torch.cuda.empty_cache()
