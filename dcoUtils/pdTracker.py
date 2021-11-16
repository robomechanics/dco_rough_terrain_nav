import torch
import sys
from wheeledSim.robotStateTransformation import robotStateTransformation

"""
This function defines the trajectory tracker controller used for our experiments.
The trajectory tracker is a simple PD controller on throttle and steering based on the
forward and lateral distance from the robot to the target position.
The inputs are the true state of the robot, the target state (from the nominal trajectory),
and optionally pd gains.
"""
def pdTracker(robotState,targetState,gains = None):
    # Gains should be provided as [kp,kd] where kp or kd is a torch tensor of gains for throttle/steering
    # use default gains if not provided
    if gains is None:
        #kp = torch.tensor([1,-3],device = targetState.device)
        #kd = torch.tensor([0.5,-1.5],device = targetState.device)
        kp = torch.tensor([1,-4],device = targetState.device)
        kd = torch.tensor([0.5,-1.5],device = targetState.device)
    else:
        kp = gains[0]
        kd = gains[1]
    oShapePrefix = targetState.shape[0:-1] # keep track of original shape so that output corresponds to input
    # get position difference of target from true (relative to true robot frame)
    robotState = robotStateTransformation(robotState)
    relXY = robotState.getRelativeState(targetState)
    if len(relXY.shape)>1:
        relXY = relXY.transpose(0,-1)[0:2,:].transpose(0,-1)    
    else:
        relXY = relXY[0:2]
    relXY = relXY.reshape(-1,relXY.shape[-1])
    # get velocity difference of target from true (relative to true robot frame)
    qTarget = targetState.reshape(-1,targetState.shape[-1])[:,3:7]
    qRobot = robotState.currentState.reshape(-1,robotState.currentState.shape[-1])[:,3:7]
    vTarget = targetState.reshape(-1,targetState.shape[-1])[:,7:10]
    vRobot = robotState.currentState.reshape(-1,robotState.currentState.shape[-1])[:,7:10]
    relVel = robotState.qrot(robotState.qinv(qRobot),robotState.qrot(qTarget,vTarget)) - vRobot
    relVelXY = relVel[:,0:2]
    # multiple position and velocity difference by gains to get action to apply
    action = relXY*kp + relVelXY*kd
    return action.reshape(oShapePrefix+(-1,))
