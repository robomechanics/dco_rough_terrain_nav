import numpy as np
import torch
import pybullet as p
import argparse
import yaml
import os
# import function that defines dynamics model
from dcoUtils.dynamicsModelNetwork import probDynModel
from dcoUtils.pdTracker import pdTracker
from dcoUtils.dco import dco,simpleGoalCost
from dcoUtils.loadSim import loadSim

if __name__ == '__main__':
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    """ load arguments for testing"""
    parser = argparse.ArgumentParser(description="Divergence Constrained Optimization in Test Environment")
    parser.add_argument('-c','--config',help='path to config directory',default='../config/')
    parser.add_argument('-m','--model_file',help='file to save trained model',default='../models/trained_model.pt')
    args = parser.parse_args()

    # set up sim
    physicsClientId = p.connect(p.GUI)
    sim_params = yaml.safe_load(open(os.path.join(args.config,'sim_params.yaml'),'r'))
    terrainParams = yaml.safe_load(open(os.path.join(args.config,'testTerrainParams.yaml'),'r'))
    sim_params.update(terrainParams)
    mapParams=[sim_params['terrainMapParams'],sim_params['senseParams']]
    sim = loadSim(sim_params)
    
    # load trained dynamics model
    model_fn = args.model_file
    model_params = yaml.safe_load(open(os.path.join(args.config,'model_params.yaml'),'r'))
    dynamicsModel = probDynModel(model_params).to(device)
    dynamicsModel.load_state_dict(torch.load(model_fn))
    
    # load test environment
    ground = np.loadtxt('rampEnv/groundZ.csv',delimiter=',')
    goalPos = np.loadtxt('rampEnv/goalPos.csv',delimiter=',')
    sim.terrain.copyGridZ(ground)
    p.addUserDebugLine([goalPos[0],goalPos[1],0],[goalPos[0],goalPos[1],10],lineColorRGB=[0,1,0],lineWidth=2,physicsClientId=sim.physicsClientId)
    sim.resetRobot()
    goalX = goalPos[0]
    goalY = goalPos[1]
    p.resetDebugVisualizerCamera(2.0,-90,-50,[0,0,1])
    
    # get starting state and terrain map
    _,newState,_ = sim.controlLoopStep([0,0])
    startState = torch.from_numpy(np.array(newState[0])).unsqueeze(0).unsqueeze(0).float().to(device)
    terrainMap = torch.from_numpy(np.array(sim.terrain.gridZ)).float().to(device)

    # load optimization parameters
    dcoParams = yaml.safe_load(open(os.path.join(args.config,'dco_params.yaml'),'r'))

    # define cost function
    cost = simpleGoalCost(goalX,goalY)
    costFcn = cost.costFcn
    optDim = [80,2]
    trackerGains=None
    nomTraj,particleTraj,actions = dco(startState,terrainMap,mapParams,dynamicsModel,optDim,dcoParams,costFcn,trackerGains=trackerGains,physicsClientId=physicsClientId)
    input('continue?')
    p.removeAllUserDebugItems(physicsClientId)
    p.addUserDebugLine([goalPos[0],goalPos[1],0],[goalPos[0],goalPos[1],10],lineColorRGB=[0,1,0],lineWidth=2,physicsClientId=sim.physicsClientId)

    # plot predictions
    for t in range(actions.shape[0]):
        p.addUserDebugLine(nomTraj[t,0:3],nomTraj[t+1,0:3],lineColorRGB=[1,0,0],lineWidth=5,physicsClientId=sim.physicsClientId)
        for i in range(particleTraj.shape[0]):
            p.addUserDebugLine(particleTraj[i,t,0:3],particleTraj[i,t+1,0:3],lineColorRGB=[1,1,0],lineWidth=1,physicsClientId=sim.physicsClientId)

    
    # simulate robot with tracker
    for t in range(actions.shape[0]):
        fbCorrection = pdTracker(torch.tensor(newState[0],device=device),nomTraj[t,:],gains=trackerGains)
        action = torch.clamp(actions[t,:]+fbCorrection,-1,1)
        stateAction,newState,terminateFlag = sim.controlLoopStep(action)
        p.addUserDebugLine(stateAction[0][0:3],newState[0][0:3],lineColorRGB=[0,1,0],lineWidth=5,physicsClientId=sim.physicsClientId)

