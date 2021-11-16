import os
import argparse
import pybullet as p
import torch
import numpy as np
import yaml
from wheeledSim.paramHandler import paramHandler
from dcoUtils.dynamicsModelNetwork import probDynModel
from dcoUtils.loadSim import loadSim
from dcoUtils.closed_loop_pred import closed_loop_pred
from dcoUtils.pdTracker import pdTracker

if __name__ == "__main__":
    """ This function tests closed-loop predictions using the trained model """
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    
    """ load arguments for testing"""
    parser = argparse.ArgumentParser(description="Test closed-loop prediction")
    parser.add_argument('-c','--config',help='path to config directory',default='../config/')
    parser.add_argument('-m','--model_file',help='file to save trained model',default='../models/trained_model.pt')
    parser.add_argument('-ol','--open_loop',help="don't use trajectory tracker/closed-loop prediction",default=False,action="store_true")
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
     
    # get starting state and terrain map
    _,newState,_ = sim.controlLoopStep([0,0])
    startState = torch.from_numpy(np.array(newState[0])).unsqueeze(0).unsqueeze(0).float().to(device)
    terrainMap = torch.from_numpy(np.array(sim.terrain.gridZ)).float().to(device)

    # use default pd tracker gains (defined in pdTracker function)
    if args.open_loop:
        kp = torch.tensor([0,0],device = device)
        kd = torch.tensor([0,0],device = device)
        trackerGains=[kp,kd]
    else:
        trackerGains = None

    # test prediction
    actions = torch.ones(1,64,2).to(device)*0.5 # action to use for trajectory prediction
    numParticles = 16 # number of particles for trajectory distribution
    nomPredTraj,particlePredTraj = closed_loop_pred(startState,actions,terrainMap,mapParams,dynamicsModel,numParticles,trackerGains=trackerGains)
    
    # plot prediction
    for t in range(actions.shape[1]):
        p.addUserDebugLine(nomPredTraj[0,t,0:3],nomPredTraj[0,t+1,0:3],lineColorRGB=[1,0,0],lineWidth=5,physicsClientId=sim.physicsClientId)
        for i in range(particlePredTraj.shape[1]):
            p.addUserDebugLine(particlePredTraj[0,i,t,0:3],particlePredTraj[0,i,t+1,0:3],lineColorRGB=[1,1,0],lineWidth=1,physicsClientId=sim.physicsClientId)

    # simulate robot with tracker
    for t in range(actions.shape[1]):
        fbCorrection = pdTracker(torch.tensor(newState[0],device=device),nomPredTraj[0,t,:],gains=trackerGains)
        action = torch.clamp(actions[0,t,:]+fbCorrection,-1,1)
        stateAction,newState,terminateFlag = sim.controlLoopStep(action)
        p.addUserDebugLine(stateAction[0][0:3],newState[0][0:3],lineColorRGB=[0,1,0],lineWidth=5,physicsClientId=sim.physicsClientId)
    input('finished?')

