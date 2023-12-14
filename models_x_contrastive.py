"""
Model architecture definitions used in WB correction.

Author: Sai Kumar Reddy Manne
"""
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch
import numpy as np
import math
import trilinear

def weights_init_normal_classifier(m):
    """
    Initialize weights of the classifier model.
    """
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.xavier_normal_(m.weight.data)

    elif classname.find("BatchNorm2d") != -1 or classname.find("InstanceNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)

##############################
#        Discriminator
##############################

def discriminator_block(in_filters, out_filters, normalization=False):
    """Returns downsampling layers of each discriminator block"""
    layers = [nn.Conv2d(in_filters, out_filters, 3, stride=2, padding=1)]
    layers.append(nn.LeakyReLU(0.2))
    if normalization:
        layers.append(nn.InstanceNorm2d(out_filters, affine=True))
        #layers.append(nn.BatchNorm2d(out_filters))

    return layers

class Classifier(nn.Module):
    """
    Classifier model for the adaptive weight prediction.
    Downsamples the image to 256x256 and predicts the AWB weights.
    """
    def __init__(self, in_channels=3):
        super(Classifier, self).__init__()

        self.model = nn.Sequential(
            nn.Upsample(size=(256,256),mode='bilinear'),
            nn.Conv2d(3, 16, 3, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.InstanceNorm2d(16, affine=True),
            *discriminator_block(16, 32, normalization=True),
            *discriminator_block(32, 64, normalization=True),
            *discriminator_block(64, 128, normalization=True),
            *discriminator_block(128, 128),
            #*discriminator_block(128, 128, normalization=True),
            nn.Dropout(p=0.5),
            nn.Conv2d(128, 3, 8, padding=0),
        )

    def forward(self, img_input):
        return self.model(img_input)

class MobileNet(nn.Module):
    # use pretrained mobilenet-v3
    """
    Pretrained MobileNet-v3 model for the adaptive weight prediction.
    Adds a linear layer to predict the AWB weights on top of MobileNet-v3.
    """
    def __init__(self, out_dim=3, aug_test=False):
        super(MobileNet, self).__init__()
        self.up = nn.Upsample(size=(224,224),mode='bilinear')
        self.model = models.mobilenet_v3_small(pretrained=True)
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(1000, out_dim))
        
    def forward(self, x):
        f = self.model(self.up(x))
        out = self.classifier(f)

        return out

class Generator3DLUT_identity(nn.Module):
    """
    Identity LUTs from generated LUT data.
    Uses Trilinear interpolation kernel from the CUDA code.
    """
    def __init__(self, dim=33):
        super(Generator3DLUT_identity, self).__init__()
        if dim == 33:
            file = open("IdentityLUT33.txt", 'r')
        elif dim == 64:
            file = open("IdentityLUT64.txt", 'r')
        lines = file.readlines()
        buffer = np.zeros((3,dim,dim,dim), dtype=np.float32)

        for i in range(0,dim):
            for j in range(0,dim):
                for k in range(0,dim):
                    n = i * dim*dim + j * dim + k
                    x = lines[n].split()
                    buffer[0,i,j,k] = float(x[0])
                    buffer[1,i,j,k] = float(x[1])
                    buffer[2,i,j,k] = float(x[2])
        self.LUT = nn.Parameter(torch.from_numpy(buffer).requires_grad_(True))
        self.TrilinearInterpolation = TrilinearInterpolation()

    def forward(self, x):
        _, output = self.TrilinearInterpolation(self.LUT, x)
        #self.LUT, output = self.TrilinearInterpolation(self.LUT, x)
        return output

class Generator3DLUT_zero(nn.Module):
    """
    Zero-basis LUTs.
    Uses Trilinear interpolation kernel from the CUDA code.
    """
    def __init__(self, dim=33):
        super(Generator3DLUT_zero, self).__init__()

        self.LUT = torch.zeros(3,dim,dim,dim, dtype=torch.float)
        self.LUT = nn.Parameter(torch.tensor(self.LUT))
        self.TrilinearInterpolation = TrilinearInterpolation()

    def forward(self, x):
        _, output = self.TrilinearInterpolation(self.LUT, x)

        return output

class TrilinearInterpolationFunction(torch.autograd.Function):
    """ Trilinear Interpolation model from the CUDA kernel"""
    @staticmethod
    def forward(ctx, lut, x):
        x = x.contiguous()

        output = x.new(x.size())
        dim = lut.size()[-1]
        shift = dim ** 3
        binsize = 1.000001 / (dim-1)
        W = x.size(2)
        H = x.size(3)
        batch = x.size(0)
        
        assert 1 == trilinear.forward(lut, 
                                      x, 
                                      output,
                                      dim, 
                                      shift, 
                                      binsize, 
                                      W, 
                                      H, 
                                      batch)

        int_package = torch.IntTensor([dim, shift, W, H, batch])
        float_package = torch.FloatTensor([binsize])
        variables = [lut, x, int_package, float_package]
        
        ctx.save_for_backward(*variables)
        
        return lut, output
    
    @staticmethod
    def backward(ctx, lut_grad, x_grad):
        
        lut, x, int_package, float_package = ctx.saved_variables
        dim, shift, W, H, batch = int_package
        dim, shift, W, H, batch = int(dim), int(shift), int(W), int(H), int(batch)
        binsize = float(float_package[0])
            
        assert 1 == trilinear.backward(x, 
                                       x_grad, 
                                       lut_grad,
                                       dim, 
                                       shift, 
                                       binsize, 
                                       W, 
                                       H, 
                                       batch)
        return lut_grad, x_grad


class TrilinearInterpolation(torch.nn.Module):
    def __init__(self):
        super(TrilinearInterpolation, self).__init__()

    def forward(self, lut, x):
        return TrilinearInterpolationFunction.apply(lut, x)


class TV_3D(nn.Module):
    """Total Variation loss for 3D LUTs"""
    def __init__(self, dim=33):
        super(TV_3D,self).__init__()

        self.weight_r = torch.ones(3,dim,dim,dim-1, dtype=torch.float)
        self.weight_r[:,:,:,(0,dim-2)] *= 2.0
        self.weight_g = torch.ones(3,dim,dim-1,dim, dtype=torch.float)
        self.weight_g[:,:,(0,dim-2),:] *= 2.0
        self.weight_b = torch.ones(3,dim-1,dim,dim, dtype=torch.float)
        self.weight_b[:,(0,dim-2),:,:] *= 2.0
        self.relu = torch.nn.ReLU()

    def forward(self, LUT):

        dif_r = LUT.LUT[:,:,:,:-1] - LUT.LUT[:,:,:,1:]
        dif_g = LUT.LUT[:,:,:-1,:] - LUT.LUT[:,:,1:,:]
        dif_b = LUT.LUT[:,:-1,:,:] - LUT.LUT[:,1:,:,:]
        tv = torch.mean(torch.mul((dif_r ** 2),self.weight_r)) + torch.mean(torch.mul((dif_g ** 2),self.weight_g)) + torch.mean(torch.mul((dif_b ** 2),self.weight_b))

        mn = torch.mean(self.relu(dif_r)) + torch.mean(self.relu(dif_g)) + torch.mean(self.relu(dif_b))

        return tv, mn


