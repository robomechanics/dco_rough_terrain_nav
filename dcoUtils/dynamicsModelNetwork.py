import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

class probDynModel(nn.Module):
    def __init__(self, model_params):
        super(probDynModel, self).__init__()
        # dimension of inputs and outputs
        self.stateDim = model_params['ioDim']['stateDim']
        self.sensingDim = model_params['ioDim']['sensingDim']
        self.actionDim = model_params['ioDim']['actionDim']
        self.predictionDim = model_params['ioDim']['predictionDim']

        # size of neural networks
        CNNSize = model_params['networkSize']['CNNSize']
        inputFCsize = model_params['networkSize']['inputFCsize']
        lstmSize = model_params['networkSize']['lstmSize']
        outputFCsize = model_params['networkSize']['outputFCsize']

        dropout = model_params['trainParams']['dropout']

        # set up CNN layers
        self.CNNlayers = []
        lastDim = torch.zeros((1,)+tuple(self.sensingDim))
        for i in range(len(CNNSize)):
            self.CNNlayers.append(nn.Conv2d(in_channels=lastDim.shape[-3],out_channels=CNNSize[i][0],
                                                kernel_size=(CNNSize[i][1],CNNSize[i][2]),
                                                stride=(CNNSize[i][3],CNNSize[i][4])))
            lastDim = self.CNNlayers[-1](lastDim)
            self.CNNlayers.append(nn.ReLU())
        self.CNNlayers = nn.Sequential(*self.CNNlayers)

        # set up input FC layers
        lastDim = lastDim.reshape(1,-1).shape[-1]+self.stateDim+self.actionDim
        self.inputFClayers = nn.ModuleList([])
        for i in range(len(inputFCsize)):
            self.inputFClayers.append(nn.Linear(lastDim,inputFCsize[i]))
            lastDim = inputFCsize[i]
            #self.inputFClayers.append(nn.ReLU())
            #self.inputFClayers.append(nn.Dropout(p=dropout))
        self.inputFClayers = nn.Sequential(*self.inputFClayers)
        self.dropout = nn.Dropout(p=dropout)

        # set up lstm
        self.lstm = nn.LSTM(input_size=lastDim,num_layers=lstmSize[0],hidden_size=lstmSize[1],batch_first = True)
        lastDim = lstmSize[1]

        # set up output FC layers
        self.outputFClayers = []
        for i in range(len(outputFCsize)):
            self.outputFClayers.append(nn.Linear(lastDim,outputFCsize[i]))
            lastDim = outputFCsize[i]
            self.outputFClayers.append(nn.ReLU())
        self.outputFClayers = nn.Sequential(*self.outputFClayers)

        self.meanFC = nn.Linear(lastDim,self.predictionDim)
        self.varFC = nn.Linear(lastDim,self.predictionDim)
    def forward(self,data,hidden=None,dropoutVals=1):
        # if dropoutVals = 1, then no dropout
        # if dropoutVals = 0, then generate new dropout
        # otherwise, use existing dropout values
        rState = data[0]
        rSense = data[1]
        rAction = data[2]
        # CNN layer
        rSense = rSense.reshape(-1,*rSense.shape[-3:])
        rSense = self.CNNlayers(rSense).reshape(*rState.shape[:-1],-1)
        # input FC layers
        connected = torch.cat((rSense,rState,rAction),dim=-1)
        #connected = self.inputFClayers(connected)
        newDropoutVals = []
        for i in range(len(self.inputFClayers)):
            connected = self.inputFClayers[i](connected)
            connected = F.relu(connected)
            newDropoutVal = self.genDropout(connected,dropoutVals,i)
            newDropoutVals.append(newDropoutVal)
            while len(newDropoutVal.shape) < len(connected.shape):
                newDropoutVal = newDropoutVal.unsqueeze(-2)
            connected = connected*newDropoutVal
        # lstm
        origShapePrefix = connected.shape[:-1]
        connected = connected.reshape(-1,origShapePrefix[-1],connected.shape[-1])
        connected,hidden = self.lstm(connected,hidden)
        connected = connected.reshape(origShapePrefix+(-1,))
        #connected = self.lstmDropout(connected)
        # output FC layers
        connected = self.outputFClayers(connected)
        mean = self.meanFC(connected)
        logVar = self.varFC(connected)
        return mean,logVar,hidden,newDropoutVals
    def genDropout(self,connected,dropoutVals,i):
        """
        if dropout val is -1, then no dropout
        if dropout val is integer other than -1, generate new dropout
            first (dropoutVals) dimensions will have different dropout values
            ex: if connected shape is (batch_size,time_steps,n), then
            dropout val = 0 won't vary dropout for different batches/ time steps
            dropout val = 1 will vary dropout for different batches, but not time steps
            dropout val = 2 will vary dropout for different batches/ time steps
        otherwise, reuse previous dropout values
        """
        if dropoutVals == -1:
            # no dropout
            return torch.ones(1,device=connected.device)
        elif type(dropoutVals) is int:
            # generate new dropout values
            dropoutShape = tuple()
            for j in range(dropoutVals):
                if j < len(connected.shape)-1:
                    dropoutShape+=(connected.shape[j],)
                else:
                    break
            dropoutShape+=(connected.shape[-1],)
            return self.dropout(torch.ones(dropoutShape,device=connected.device))
        else:
            # reuse old dropout values
            return dropoutVals[i]
