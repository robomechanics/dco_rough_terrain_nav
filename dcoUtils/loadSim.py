from wheeledRobots.clifford.cliffordRobot import Clifford
from wheeledSim.simController import simController

def loadSim(sim_params,physicsClientId=0):
    robotParams = {}
    simParams = {}
    terrainMapParams = {}
    terrainParams = {}
    senseParams = {}
    explorationParams = {}
    
    robotParams.update(sim_params['robotParams'])
    simParams.update(sim_params['simParams'])
    terrainMapParams.update(sim_params['terrainMapParams'])
    terrainParams.update(sim_params['terrainParams'])
    senseParams.update(sim_params['senseParams'])
    
    robot = Clifford(params=robotParams,physicsClientId=physicsClientId)
    sim = simController(robot,simulationParamsIn=simParams,senseParamsIn=senseParams,terrainMapParamsIn=terrainMapParams,
        terrainParamsIn=terrainParams,physicsClientId=physicsClientId)
    return sim
