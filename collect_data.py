import os
import time
import torch
from dcoUtils.loadSim import loadSim

class singleProcess:
    def __init__(self,index):
        self.index = index
    def setup(self,numTrajectories,trajectoryLength,dataDir,sim_params):
        print("setup sim " +str(self.index))
        self.trajectoryLength = trajectoryLength
        self.numTrajectories = numTrajectories
        self.dataDir = dataDir
        # start sim
        physicsClientId = p.connect(p.DIRECT)
        self.sim = loadSim(sim_params,physicsClientId)
        self.fileCounter = 0
        self.filenames = []
        self.trajectoryLengths = []
    def newTrajectoryData(self,stateAction,newState):
        self.trajectoryData = []
        for i in range(len(stateAction)):
            self.trajectoryData.append(torch.from_numpy(np.array(stateAction[i])).unsqueeze(0).float())
        for i in range(len(newState)):
            self.trajectoryData.append(torch.from_numpy(np.array(newState[i])).unsqueeze(0).float())
        self.trajectoryData.append(torch.from_numpy(np.array(self.sim.terrain.gridZ)).float())
    def addSampleToTrajData(self,stateAction,newState):
        for i in range(len(stateAction)):
            self.trajectoryData[i] = torch.cat((self.trajectoryData[i],torch.from_numpy(np.array(stateAction[i])).unsqueeze(0).float()),dim=0)
        for i in range(len(newState)):
            self.trajectoryData[i+len(stateAction)] = torch.cat((self.trajectoryData[i+len(stateAction)],torch.from_numpy(np.array(newState[i])).unsqueeze(0).float()),dim=0)
    def saveTrajectory(self):
        filename = 'sim'+str(self.index)+'_'+str(self.fileCounter)+'.pt'
        while os.path.exists(os.path.join(self.dataDir,filename)):#self.dataDir+filename):
            self.fileCounter+=1
            filename = 'sim'+str(self.index)+'_'+str(self.fileCounter)+'.pt'
        self.filenames.append(filename)
        self.trajectoryLengths.append(self.trajectoryData[0].shape[0])
        torch.save(self.trajectoryData,os.path.join(self.dataDir,filename))#self.dataDir+filename)
    def gatherSimData(self):
        sTime = time.time()
        while len(self.filenames) < self.numTrajectories:
            # while haven't gathered enough data
            # reset simulation start new trajectory
            self.sim.newTerrain()
            self.sim.resetRobot()
            stateAction,newState,terminateFlag = self.sim.controlLoopStep(self.sim.randomDriveAction())
            self.newTrajectoryData(stateAction,newState)
            while not terminateFlag:
                # while robot isn't stuck, step simulation and add data
                stateAction,newState,terminateFlag = self.sim.controlLoopStep(self.sim.randomDriveAction())
                self.addSampleToTrajData(stateAction,newState)
                if self.trajectoryData[0].shape[0] >= self.trajectoryLength:
                    # if trajectory is long enough, save trajectory and start new one
                    self.saveTrajectory()
                    break
            # print estimated time left
            if len(self.filenames) > 0:
                runTime = (time.time()-sTime)/3600
                print("sim: " + str(self.index) + ", numTrajectories: " + str(len(self.filenames)) + ", " + 
                        "time elapsed: " + "%.2f"%runTime + " hours, " + 
                        "estimated time left: " + "%.2f"%(float(self.numTrajectories-len(self.filenames))*runTime/float(len(self.filenames))) + "hours")
        return self.filenames,self.trajectoryLengths
    def outputIndex(self):
        return self.index

if __name__ == '__main__':
    import sys
    import argparse
    import yaml
    import pybullet as p
    import numpy as np
    import concurrent.futures
    import csv
    parser = argparse.ArgumentParser(description="Collects trajectory data for training")
    parser.add_argument('-c','--config',help='path to config directory',default='config/')
    parser.add_argument('-d','--data_dir',help='path to training data directory',default='training_data/')
    args = parser.parse_args()
    
    # parameters for parallel processing
    sim_params = yaml.safe_load(open(os.path.join(args.config,'sim_params.yaml'),'r'))
    terrainParams = yaml.safe_load(open(os.path.join(args.config,'trainingTerrainParams.yaml'),'r'))
    sim_params.update(terrainParams)
    numParallelSims = sim_params['dataCollectParams']['numParallelSims']
    trajectoryLength = sim_params['dataCollectParams']['trajectoryLength']
    totalNumTrajectories = sim_params['dataCollectParams']['totalNumTrajectories']
    numTrajsPerSim = int(np.floor(totalNumTrajectories/numParallelSims))
    
    # create data directory if it doesn't exist
    if not os.path.isdir(args.data_dir):
        os.mkdir(args.data_dir)
    
    # initialize all parallel sims
    processes = [singleProcess(i) for i in range(numParallelSims)]
    for process in processes:
        process.setup(numTrajsPerSim,trajectoryLength,args.data_dir,sim_params)
    processes[-1].numTrajectories = totalNumTrajectories-(numParallelSims-1)*numTrajsPerSim
    print("finished initialization")
    
    # start collecting data
    executor = concurrent.futures.ProcessPoolExecutor()
    results = [executor.submit(process.gatherSimData) for process in processes]
    concurrent.futures.wait(results,return_when=concurrent.futures.ALL_COMPLETED)
    
    # write metadata csv file
    csvFile = os.path.join(args.data_dir,'meta.csv')
    startNewFile = False
    if startNewFile:
        csvFile = open(csvFile, 'w', newline='')
    else:
        csvFile = open(csvFile, 'a', newline='')
    csvWriter = csv.writer(csvFile,delimiter=',')
    if startNewFile:
        csvWriter.writerow(['filenames','trajectoryLengths'])
    for result in results:
        fileNames = result.result()[0]
        trajLengths = result.result()[1]
        for i in range(len(fileNames)):
            csvWriter.writerow([fileNames[i],trajLengths[i]])
    csvFile.flush()
    for process in processes:
        p.disconnect(process.sim.physicsClientId)
