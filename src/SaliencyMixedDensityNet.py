import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import numpy 
import torch.optim as optim

import torchvision.models as models
from salicon_model import Salicon

from PIL import Image
from torchray.utils import imsc
from torchvision import transforms
import cv2

class SaliencyMixedDensityNetwork(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, saliency_size, init_w=3e-3, std_min=1e-10, std_max=1):
        super(SaliencyMixedDensityNetwork, self).__init__()
        """
            The model computes the Saliency Mixture Density Network
            the inputs are:
            input_size : (int) dimension of input features
            output_size : (int) dimension of outputs
            hideen_size : (int) width of hidden layers
            saliency_size : (int) dimension of image input
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

        # Input layer for head movement feature
        self.linear1 = nn.Linear(input_size,hidden_size)

        # Input layer for saliency feature
        self.linearimg = nn.Linear(saliency_size,hidden_size)

        # Define hidden layers
        self.linear2 = nn.Linear(hidden_size*2, hidden_size)
        self.linear3 = nn.Linear(hidden_size, hidden_size)
        self.linear4 = nn.Linear(hidden_size, hidden_size)

        
        # Define Batch Normalization layer
        self.bn1 = nn.BatchNorm1d(hidden_size*2)
        
        # Define activation function for FC layers
        self.leakyrelu = nn.LeakyReLU(0.0001)
        
        # define the output layers for mean and standard deviation as well as mixing coefficient w
        self.mean_linearX = nn.Linear(hidden_size, output_size)
        self.mean_linearY = nn.Linear(hidden_size, output_size)


        self.std_linearX = nn.Linear(hidden_size, output_size)
        self.std_linearY = nn.Linear(hidden_size, output_size)


        self.weight_linear = nn.Linear(hidden_size, output_size)
        
        self.loss_criteria=nn.MSELoss()

    def forward(self, state, saliency):
        """the function computes the output by forward propagation through model
            inputs: 
                    state : array of floats with size = (n,5)
                    saliency : array of floats with size = (n,144*128)
            return:
                    means : array of torch tensors with shape (n,2,5)
                    stds : array of torch tensors with shape (n,2,5)
                    weights : array of torch tensors with shape (n,1,5)

        """
        
        # convert the head movement input to tensor of 256 dimensions
        x = self.leakyrelu(self.linear1(state))

        # convert the saliency map input to tensor of 256 dimensions
        ximg = self.leakyrelu(self.linearimg(saliency))

        # concatenent two two features to produce new feature of 512 dimensions
        integrated = torch.cat((x,ximg), 1)

        # forward propagate through hidden layers
        x = self.leakyrelu(self.linear2(self.bn1(integrated)))
        x = self.leakyrelu(self.linear3(x))
        x = self.leakyrelu(self.linear4(x))

        # compute outputs from output layer
        meanX    = torch.sigmoid(self.mean_linearX(x)).reshape(-1,1,self.output_size)
        meanY    = torch.sigmoid(self.mean_linearY(x)).reshape(-1,1,self.output_size)
        stdX    = torch.sigmoid(self.std_linearX(x)).reshape(-1,1,self.output_size)
        stdY    = torch.sigmoid(self.std_linearY(x)).reshape(-1,1,self.output_size)
        weights = F.softmax(self.weight_linear(x),dim=1).reshape(-1,self.output_size,1)

        
        return torch.cat((meanX,meanY),1) , torch.cat((stdX,stdY),1), weights

    def compute_loss(self,state,saliency,targets):
        """the function computes the loss from model outputs
            inputs: 
                    state : torch tensors with size = (n,5)
                    saliency : torch tensors with size = (n,144*128)
                    targets : torch tensors with size = (n,2)
            returns:
                    loss : torch tensor (1,1) 

        """

        # reshape targets to match the dimension of means
        reshaped_targets = targets.reshape(-1,2,1)
        for i in range(self.output_size-1):
            reshaped_targets = torch.cat((reshaped_targets,targets.reshape(-1,2,1)),2)

        # compute output distribution from model
        means,stds,weights = self.forward(state,saliency)

        # compute the mixed log pdf
        mixed_pdf = mixed_prob(means,stds,weights,reshaped_targets)

        # make sure there is no 0 standard deviation produced which leads to abnormal gradiant
        checker = (mixed_pdf != torch.zeros(mixed_pdf.shape).to('cuda'))
        mixed_pdf = mixed_pdf.where(checker, torch.tensor(0.0000001).to('cuda'))

        # compute the average negative log likelihood
        return -torch.log(mixed_pdf).mean()
def mixed_prob(means,stds,weights,validt):
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




def compute_saliencymap(image_path,input_feature,model_weight = None):
    """the function constructs the pipeline for the training and prediction of SMDN model
        inputs: 
                image_path : string the path of image
                input_feature : torch tensor with size = (1,2) the head position in Equirectangular coordinate
                model_weight : string the path of model weights
        returns:     
    """  
    # Initialize model 
    model = SaliencyMixedDensityNetwork()

    # load model weights
    if model_weight != None:
        model.load_state_dict(torch.load(model_weight))
    # assign to device
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    # create coarse and fine image
    img_tensor_coarse = image2tensor(image_path,center = (input_feature[0,0],input_feature[0,1])).to(device)
    img_tensor_fine = image2tensor(image_path,resolution = 2,center = (input_feature[0,0],input_feature[0,1])).to(device)
    
    # compute saliency_map
    saliency_map = model(img_tensor_fine,img_tensor_coarse).to(device)
    pred_image=saliency_map.squeeze()

    # add Gaussain mask to the image
    smap=(pred_image-torch.min(pred_image))/((torch.max(pred_image)-torch.min(pred_image)))
    smap=smap.detach().cpu().numpy()
    smap=cv2.resize(smap,(144,128),interpolation=cv2.INTER_CUBIC)
    smap=cv2.GaussianBlur(smap,(75,75),8,cv2.BORDER_DEFAULT)
    return smap

def convert_smap2tensor_simple(smap,kernel_size =(10,10)):
    """the function converts 4D sleincy map images to 2D tensors
        inputs: 
                smap : saliency map
                kernel_size : size of pooling window
        returns:
                torch tenosr of the pooled saliency map 
    """  
    pooling = nn.MaxPool2d(kernel_size, stride=(2, 1))
    return pooling(smap)


def image2tensor(image_path,resolution = 1,center = (1920,960)):
    # convert image to torch tensor with shape (1 * 3 * 224 * 224)
    img = Image.open(image_path)

    x, y = center
    img = img.crop((x - 720, y - 640, x+720, y + 640))

    #p = transforms.Compose([transforms.Scale((3840, 1920))])


    fine = resolution
    fine *= 3
    data_transforms = transforms.Compose([
            transforms.Resize((144*fine,128*fine),interpolation=0),
            transforms.ToTensor(),
            transforms.Normalize(mean=vgg16_mean, std=vgg16_std),])

    img, i = imsc(data_transforms(img), quiet=False)
    img = img.to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
    return torch.reshape(img, (1, 3, 144*fine, 128*fine))
