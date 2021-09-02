import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import numpy 
import torch.optim as optim
class MixtureDensityNetwork(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, init_w=3e-3, std_min=1e-10, std_max=1):
        super(MixtureDensityNetwork, self).__init__()
        """
            The model computes the Mixture Density Network
            the inputs are:
            input_size : (int) dimension of input features
            output_size : (int) dimension of outputs
            hideen_size : (int) width of hidden layers
            init_w : optional (float) initialized weights
            std_min : optional (float) the lower bound of standard deviation
            std_max : optional (float) the upper bound of standard deviation
        """
        # Define hyperparameters
        self.std_min = std_min
        self.std_max = std_max
        self.hidden_dim = hidden_size
        self.output_size = output_size

        # Define FC layers
        self.linear1 = nn.Linear(input_size,hidden_size)

        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, hidden_size)


        # Define Batch Normalization layer
        self.bn1 = nn.BatchNorm1d(hidden_size)

        
        # Define activation function for FC layers
        self.leakyrelu = nn.LeakyReLU(0.1)
        
        # define the output layers for mean and standard deviation as well as mixing coefficient w
        self.mean_linearX = nn.Linear(hidden_size, output_size)
        self.mean_linearY = nn.Linear(hidden_size, output_size)


        self.std_linearX = nn.Linear(hidden_size, output_size)
        self.std_linearY = nn.Linear(hidden_size, output_size)


        self.weight_linear = nn.Linear(hidden_size, output_size)


        self.loss_criteria = nn.MSELoss()
    def forward(self, state):
        """the function computes the output by forward propagation through model
            inputs: 
                    state : torch tensors with size = (n,5)
            return:
                    means : torch tensors with shape (n,2,5)
                    stds : torch tensors with shape (n,2,5)
                    weights : torch tensors with shape (n,1,5)

        """

        # forward propagate through hidden layers
        x = self.leakyrelu(self.linear1(state))
        x = self.leakyrelu(self.linear2(x))

        # compute outputs from output layer
        meanX    = torch.sigmoid(self.mean_linearX(x)).reshape(-1,1,self.output_size)
        meanY    = torch.sigmoid(self.mean_linearY(x)).reshape(-1,1,self.output_size)
        stdX    = torch.sigmoid(self.std_linearX(x)).reshape(-1,1,self.output_size)
        stdY    = torch.sigmoid(self.std_linearY(x)).reshape(-1,1,self.output_size)
        weights = F.softmax(self.weight_linear(x),dim=1).reshape(-1,self.output_size,1)

 
        return torch.cat((meanX,meanY),1) , torch.cat((stdX,stdY),1), weights

    def compute_loss(self,state,targets):
        """the function computes the loss from model outputs
            inputs: 
                    state : torch tensors with size = (n,5)
                    targets : torch tensors with size = (n,2)
            returns:
                    loss : torch tensor (1,1) 
        """
        reshaped_targets = targets.reshape(-1,2,1)

        # reshape targets to match the dimension of means
        for i in range(self.output_size-1):
            reshaped_targets = torch.cat((reshaped_targets,targets.reshape(-1,2,1)),2)

        # compute output distribution from model
        means,stds,weights = self.forward(state)

        # compute the mixed log pdf
        mixed_pdf = mixed_prob(means,stds,weights,reshaped_targets)

        # compute the average negative log likelihood
        return -torch.log(mixed_pdf).mean()

def mixed_prob( means,stds,weights,validt):
        """the function computes the mixed log pdf of dirstributions
            inputs: 
                    means : torch tensor with size = (n,2,5)
                    stds : torch tensor with size = (n,2,5)
                    weights : torch tensor with size = (n,1,5)
                    validt : torch tensor with size = (n,2,5)
            returns:
                    matrix of log pdf  : torch tensors with size = (n,1,5)
        """
    prob = Normal(means,stds).log_prob(validt).exp()
    return torch.matmul(prob,weights).reshape(-1,2)