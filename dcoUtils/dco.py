import torch
from torch.optim import Adam,SGD,RMSprop
from dcoUtils.closed_loop_pred import closed_loop_pred
import pybullet as p

def randActionInit(batchSize,Time,dim):
    randMean = torch.zeros(batchSize,1,dim)
    randMean[:,:,0] = 0.25
    randStd = torch.ones(batchSize,1,dim)
    randStd[:,:,1] = 3
    actions = torch.normal(randMean,randStd)
    for t in range(Time-1):
        newActions = 0.8*actions[:,-1:,:]+0.25*torch.normal(randMean,randStd)
        actions = torch.cat((actions,newActions),dim=1)
    return actions

class simpleGoalCost:
    def __init__(self,goalX,goalY):
        self.goalX = goalX
        self.goalY = goalY
    def costFcn(self,actions,predTraj):
        cost = (predTraj[:,-1,0]-self.goalX)**2 + (predTraj[:,-1,1]-self.goalY)**2
        return cost


def divergenceCost(nomTraj,particleTraj):
    particleDiff = particleTraj - nomTraj.unsqueeze(1)
    #divergenceScale = torch.tensor(self.divergenceScale,device=nomTraj.device)
    #while len(divergenceScale.shape) < len(particleDiff.shape):
    #    divergenceScale = divergenceScale.unsqueeze(0)
    divergenceScale = 1
    particleScaledDiff = particleDiff*divergenceScale
    divergenceCost = particleScaledDiff**2
    divergenceCost = divergenceCost.mean(dim=-1) # average across state dim
    divergenceCost = divergenceCost.max(dim=-1)[0] # max across time step
    divergenceCost = divergenceCost.mean(dim=-1) # average across particles
    return divergenceCost

def dco(startState,terrainMap,mapParams,dynamicsModel,optDim,dcoParams,costFunction,trackerGains=None,physicsClientId=None):
    """
    This function performs divergence constrained optimization
    Inputs:
    startState: current state of robot
    terrainMap: map of terrain
    mapParams: parameters of map/sensing
    dynamics model: predictive model to use
    optDim: action space to optimize (timesteps,action_dim)
    dcoParams: optimization params
    costFunction: function that inputs actions and pred trajectories and outputs cost
    trackerGains: optional gains for trajectory tracking controller
    """
    
    """ initialize particles (These are atanh of action trajectories) """
    #actionMean = torch.zeros(dcoParams['batchSize'],optDim[0],optDim[1],device=startState.device)
    #actionMean[:,:,0] = 0.5
    #actionSTD = torch.ones_like(actionMean)
    #actionSTD[:,:,0] = 0.1
    #actions = torch.normal(actionMean,actionSTD)
    #actions = randActionInit(5*dcoParams['batchSize'],optDim[0],optDim[1]).to(startState.device).clamp(-1,1)
    #actions = (torch.tensor([2,0]).unsqueeze(0).unsqueeze(0)+torch.rand + torch.rand(dcoParams['batchSize'],optDim[0],optDim[1])).to(startState.device)
    plotLines = []

    """ Try out a bunch of actions and keep the best """
    actions = torch.tensor([],device=startState.device)
    actionCosts = torch.tensor([],device=startState.device)
    actionNomTraj = torch.tensor([],device=startState.device)
    print('generating random actions')
    with torch.no_grad():
        for i in range(50):
            randActions = randActionInit(dcoParams['batchSize'],optDim[0],optDim[1]).to(startState.device).clamp(-1,1)
            nomTraj,_ = closed_loop_pred(startState,torch.tanh(randActions),terrainMap,mapParams,dynamicsModel,0,trackerGains=trackerGains)
            traj_costs = costFunction(torch.tanh(actions),nomTraj)
            actions = torch.cat((actions,randActions),dim=0).detach()
            actionCosts = torch.cat((actionCosts,traj_costs),dim=0).detach()
            actionNomTraj = torch.cat((actionNomTraj,nomTraj),dim=0).detach()
            _,indices = actionCosts.sort()
            actions = actions[indices[0:dcoParams['batchSize']],:]
            actionCosts = actionCosts[indices[0:dcoParams['batchSize']]]
            actionNomTraj = actionNomTraj[indices[0:dcoParams['batchSize']]]
            # do plotting
            if not physicsClientId is None:
                index = 0
                for j in range(actionNomTraj.shape[0]):
                    lineColor = [0,1,0] if j==0 else [1,1,0]
                    for t in range(actionNomTraj.shape[1]-1):
                        if i==0:
                            plotLines.append(p.addUserDebugLine(actionNomTraj[j,t, 0:3],actionNomTraj[j,t+1, 0:3],lineColorRGB=lineColor,lineWidth=2,physicsClientId=physicsClientId))
                        else:
                            p.addUserDebugLine(actionNomTraj[j,t, 0:3],actionNomTraj[j,t+1, 0:3],lineColorRGB=lineColor,lineWidth=2,physicsClientId=physicsClientId,replaceItemUniqueId=plotLines[index])
                            index+=1


    
    actions.requires_grad=True
    optimizer = Adam([actions],lr=dcoParams['stepSize'])#,weight_decay=0.5)

    
    """ Repeatedly optimize, while increasing cost of constraint violation """
    for lambda_idx in range(len(dcoParams["lambdas"])):
        _lam = dcoParams["lambdas"][lambda_idx]
        print('lambda: '+ str(_lam))
        # update particles
        for i in range(dcoParams["iterations"][lambda_idx]):
            print('---- opt iteration: ' + str(i + 1))
            # simulate actions
            nomTraj,particleTraj = closed_loop_pred(startState,torch.tanh(actions),terrainMap,mapParams,dynamicsModel,dcoParams['numPredParticles'],trackerGains=trackerGains)
            traj_costs = costFunction(torch.tanh(actions),nomTraj)
            divergences = divergenceCost(nomTraj,particleTraj)
            costs = traj_costs + _lam * (divergences - dcoParams['divergenceUB']).clamp(0.)
            _,costOrder = costs.sort()
            print(costOrder)
            print('traj_costs: ' + str([round(item,2) for item in traj_costs.tolist()]))
            print('divergence: ' + str([round(item,2) for item in divergences.tolist()]))
            print('total: ' + str([round(item,2) for item in costs.tolist()]))
            optimizer.zero_grad()
            costs.sum().backward()
            optimizer.step()
            torch.cuda.empty_cache()
            if not physicsClientId is None:
                index = 0
                for j in range(nomTraj.shape[0]):
                    lineColor = [0,1,0] if j == costOrder[0] else [1,1,0]
                    for t in range(nomTraj.shape[1]-1):
                        p.addUserDebugLine(nomTraj[j,t, 0:3],nomTraj[j,t+1, 0:3],lineColorRGB=lineColor,lineWidth=2,physicsClientId=physicsClientId,replaceItemUniqueId=plotLines[index])
                        index+=1
        bestNomTraj = nomTraj[costs.argmin(),:]
        bestParticleTraj = particleTraj[costs.argmin(),:]
        bestAction = torch.tanh(actions[costs.argmin(),:])

    return bestNomTraj,bestParticleTraj,bestAction
#                if noDecreaseCount > self.optParams['maxNoDecrease']:
#                    break

"""
        #Visualize the nominal traj after each AL iter:
        currDivergence = (currParticleTraj - currNomTraj.unsqueeze(0)).pow(2).mean(dim=[0, 2]) #Divergence is avg deviation across time/features.
        for t in range(currNomTraj.shape[0]-1):
#                c = (i_al+1)/self.optParams["ALmaxIterations"]
#                p.addUserDebugLine(currNomTraj[t, 0:3],currNomTraj[t+1, 0:3],lineColorRGB=[c,0,1-c],lineWidth=2,physicsClientId=self.sim.physicsClientId)

            c = min(currDivergence[t].sqrt(), 1.)
            p.addUserDebugLine(currNomTraj[t, 0:3],currNomTraj[t+1, 0:3],lineColorRGB=[c,0,1-c],lineWidth=2,physicsClientId=self.sim.physicsClientId)
        #p.addUserDebugText('{}'.format(i_al+1), currNomTraj[-20, 0:3])
"""
