import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import numpy 
import torch.optim as optim
class BaseNetwork(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, init_w=3e-3, std_min=1e-10, std_max=1):
        super(BaseNetwork, self).__init__()
        """
            The model computes a Deep Neural Network
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
        
        # Define FC layers
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, hidden_size)
        self.linear4 = nn.Linear(hidden_size, hidden_size)

        # Define Batch Normalization layer
        self.bn1 = nn.BatchNorm1d(input_size)
        self.bn2 = nn.BatchNorm1d(hidden_size)
        self.bn3 = nn.BatchNorm1d(hidden_size)
        self.bn4 = nn.BatchNorm1d(hidden_size)
        
        # Define activation function for FC layers
        self.leakyrelu = nn.LeakyReLU(0.1)
        
        # define the output layers for mean and standard deviation
        self.mean_linear = nn.Linear(hidden_size, output_size)
        self.mean_linear.weight.data.uniform_(-init_w, init_w)
        self.mean_linear.bias.data.uniform_(-init_w, init_w)

        self.std_linear = nn.Linear(hidden_size, output_size)
        self.std_linear.weight.data.uniform_(-init_w, init_w)
        self.std_linear.bias.data.uniform_(-init_w, init_w)
        
    def forward(self, state):
        """the function computes the output by forward propagation through model
            inputs: 
                    state : torch tensors with size = (n,5)
            return:
                    means : torch tensors with shape (n,2)
                    stds : torch tensors with shape (n,2)
        """

        # forward propagate through hidden layers
        x = self.leakyrelu(self.linear1(self.bn1(state)))
        x = self.leakyrelu(self.linear2(self.bn2(x)))
        
        # compute outputs from output layer
        mean    = F.sigmoid(self.mean_linear(self.bn3(x)))
        std    = F.sigmoid(self.std_linear(self.bn4(x)))
        
        std = torch.clamp(std, self.std_min, self.std_max)
        
        return mean,std
    def get_loss(self,state,target):
        """the function computes loss of model
            inputs: 
                    state : torch tensors with size = (n,5)
                    target : torch tensors with size = (n,2)
            return:
                    negative log likelihood : torch tensor (1,1)
        """
        # computes mean and standard deviation
        mean,std = self.forward(state)

        # construct distribution and compute log probability density function
        log_pdf = Normal(mean,std).log_prob(target)
        pdf = log_pdf.exp()
        return -pdf.mean()
