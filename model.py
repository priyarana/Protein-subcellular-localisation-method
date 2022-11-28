
from torchvision import models
from torchvision.models.resnet import ResNet, Bottleneck, BasicBlock, model_urls, resnet34
from pretrainedmodels.models import bninception
import pretrainedmodels
from torch import nn
from config import config
from collections import OrderedDict
import torch.nn.functional as F
import torch
import random
import pretrainedmodels
from utils import mixup_data
from utils import per_image_standardization
import os
import torch.nn as nn
import torch.nn.init as init
from .common import conv1x1_block, SEBlock
from .resnet import ResInitBlock
from .resnext import ResNeXtBottleneck

def get_net():
    model = bninception(pretrained="imagenet")
    model.global_pool = nn.AdaptiveAvgPool2d(1)
    model.conv1_7x7_s2 = nn.Conv2d(config.channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3))
    model.last_linear = nn.Sequential(
                nn.BatchNorm1d(1024),
                nn.Dropout(0.5),
                nn.Linear(1024, config.num_classes)
            )
    return model

def get_resnet34(num_classes=28, **_):
    model_name = 'resnet34'
    model = pretrainedmodels.__dict__[model_name](num_classes=1000, pretrained='imagenet')
    conv1 = model.conv1
    model.conv1 = nn.Conv2d(in_channels=4,
                            out_channels=conv1.out_channels,
                            kernel_size=conv1.kernel_size,
                            stride=conv1.stride,
                            padding=conv1.padding,
                            bias=conv1.bias)
    # copy pretrained weights
    model.conv1.weight.data[:,:3,:,:] = conv1.weight.data
    model.conv1.weight.data[:,3:,:,:] = conv1.weight.data[:,:1,:,:]
    
    model.global_pool = nn.AdaptiveAvgPool2d(1)
    model.last_linear = nn.Sequential(
                nn.BatchNorm1d(512),
                nn.Dropout(0.5),
                nn.Linear(512, 28))
    return model

def get_SEResNext50(num_classes=28, **_):
    model_name = 'se_resnext50_32x4d'   #se_resnext50_32x4d(num_classes=1000, pretrained='imagenet')
    
    #model = torch.hub.load('pytorch/vision', 'resnext50_32x4d', pretrained=True)
    model = pretrainedmodels.__dict__[model_name](num_classes=1000, pretrained='imagenet')
    #print(model)
    #conv1 = model.layer0
    model.layer0.conv1 = nn.Conv2d(in_channels=4,out_channels= 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    
    '''nn.Conv2d(in_channels=4,
                            out_channels=layer0.conv1.out_channels,
                            kernel_size=conv1.kernel_size,
                            stride=conv1.stride,
                            padding=conv1.padding,
                            bias=conv1.bias)'''
    # copy pretrained weights
    model.layer0.conv1.weight.data[:,:4,:,:] = model.layer0.conv1.weight.data
    model.layer0.conv1.weight.data[:,4:,:,:] = model.layer0.conv1.weight.data[:,:1,:,:]
    
    model.global_pool = nn.AdaptiveAvgPool2d(1)
    model.last_linear = nn.Sequential(
                nn.BatchNorm1d(204800),
                nn.Dropout(0.5),
                nn.Linear(204800, 28))
    return model
    
    
    
def get_resnet50(num_classes=28, **_):
    model_name = 'resnet50'   #se_resnext50_32x4d(num_classes=1000, pretrained='imagenet')

    model = pretrainedmodels.__dict__[model_name](num_classes=1000, pretrained='imagenet')

    conv1 = model.conv1  #conv1): Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    model.conv1 = nn.Conv2d(in_channels=4,
                            out_channels=conv1.out_channels,
                            kernel_size=conv1.kernel_size,
                            stride=conv1.stride,
                            padding=conv1.padding,
                            bias=conv1.bias)
    # copy pretrained weights
    model.conv1.weight.data[:,:3,:,:] = conv1.weight.data
    model.conv1.weight.data[:,3:,:,:] = conv1.weight.data[:,:1,:,:]
    
    model.global_pool = nn.AdaptiveAvgPool2d(1)
    model.last_linear = nn.Sequential(
                nn.BatchNorm1d(2048),
                nn.Dropout(0.5),
                nn.Linear(2048, 28))
    return model
    
class SEResNeXtUnit(nn.Module):
    """
    SE-ResNeXt unit.
    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    stride : int or tuple/list of 2 int
        Strides of the convolution.
    cardinality: int
        Number of groups.
    bottleneck_width: int
        Width of bottleneck block.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 stride,
                 cardinality,
                 bottleneck_width):
        super(SEResNeXtUnit, self).__init__()
        self.resize_identity = (in_channels != out_channels) or (stride != 1)

        self.body = ResNeXtBottleneck(
            in_channels=in_channels,
            out_channels=out_channels,
            stride=stride,
            cardinality=cardinality,
            bottleneck_width=bottleneck_width)
        self.se = SEBlock(channels=out_channels)
        if self.resize_identity:
            self.identity_conv = conv1x1_block(
                in_channels=in_channels,
                out_channels=out_channels,
                stride=stride,
                activation=None)
        self.activ = nn.ReLU(inplace=True)

    def forward(self, x):
        if self.resize_identity:
            identity = self.identity_conv(x)
        else:
            identity = x
        x = self.body(x)
        x = self.se(x)
        x = x + identity
        x = self.activ(x)
        return x


class SEResNeXt(nn.Module):
    """
    SE-ResNeXt model from 'Squeeze-and-Excitation Networks,' https://arxiv.org/abs/1709.01507.
    Parameters:
    ----------
    channels : list of list of int
        Number of output channels for each unit.
    init_block_channels : int
        Number of output channels for the initial unit.
    cardinality: int
        Number of groups.
    bottleneck_width: int
        Width of bottleneck block.
    in_channels : int, default 3
        Number of input channels.
    in_size : tuple of two ints, default (224, 224)
        Spatial size of the expected input image.
    num_classes : int, default 1000
        Number of classification classes.
    """
    def __init__(self,
                 channels,
                 init_block_channels,
                 cardinality,
                 bottleneck_width,
                 in_channels=4,
                 in_size=(512, 512),
                 num_classes=28):
        super(SEResNeXt, self).__init__()
        self.in_size = in_size
        self.num_classes = 28

        self.features = nn.Sequential()
        self.features.add_module("init_block", ResInitBlock(
            in_channels=in_channels,
            out_channels=init_block_channels))
        in_channels = init_block_channels
        for i, channels_per_stage in enumerate(channels):
            stage = nn.Sequential()
            for j, out_channels in enumerate(channels_per_stage):
                stride = 2 if (j == 0) and (i != 0) else 1
                stage.add_module("unit{}".format(j + 1), SEResNeXtUnit(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    stride=stride,
                    cardinality=cardinality,
                    bottleneck_width=bottleneck_width))
                in_channels = out_channels
            self.features.add_module("stage{}".format(i + 1), stage)
        self.features.add_module("final_pool", nn.AvgPool2d(
            kernel_size=7,
            stride=1))
            
        self.output = nn.Sequential(
                nn.Dropout(0.5),
                nn.Linear(204800, 28))

        '''self.output = nn.Linear(
            in_features=in_channels*100,
            out_features=num_classes)'''

        self._init_params()

    def _init_params(self):
        for name, module in self.named_modules():
            if isinstance(module, nn.Conv2d):
                init.kaiming_uniform_(module.weight)
                if module.bias is not None:
                    init.constant_(module.bias, 0)

    def forward(self, x):
        
        x = self.features(x)
        
        x = x.view(x.size(0), -1)
        
        x = self.output(x)
        print('4-->', x.size())
        return x


def get_seresnext(blocks,
                  cardinality,
                  bottleneck_width,
                  model_name=None,
                  pretrained=True,
                  root=os.path.join("~", ".torch", "models"),
                  **kwargs):
    """
    Create SE-ResNeXt model with specific parameters.
    Parameters:
    ----------
    blocks : int
        Number of blocks.
    cardinality: int
        Number of groups.
    bottleneck_width: int
        Width of bottleneck block.
    model_name : str or None, default None
        Model name for loading pretrained model.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """

    if blocks == 50:
        layers = [3, 4, 6, 3]
    elif blocks == 101:
        layers = [3, 4, 23, 3]
    else:
        raise ValueError("Unsupported SE-ResNeXt with number of blocks: {}".format(blocks))

    init_block_channels = 64
    channels_per_layers = [256, 512, 1024, 2048]

    channels = [[ci] * li for (ci, li) in zip(channels_per_layers, layers)]

    net = SEResNeXt(
        channels=channels,
        init_block_channels=init_block_channels,
        cardinality=cardinality,
        bottleneck_width=bottleneck_width,
        **kwargs)

    if pretrained:
        if (model_name is None) or (not model_name):
            raise ValueError("Parameter `model_name` should be properly initialized for loading pretrained model.")
        from .model_store import download_model
        download_model(
            net=net,
            model_name=model_name,
            local_model_store_dir_path='/home/561/pr2894/.cache/torch/hub/checkpoints/')

    return net


def seresnext50_32x4d(**kwargs):
    """
    SE-ResNeXt-50 (32x4d) model from 'Squeeze-and-Excitation Networks,' https://arxiv.org/abs/1709.01507.
    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    return get_seresnext(blocks=50, cardinality=32, bottleneck_width=4, model_name="seresnext50_32x4d", **kwargs)

      
class CustomResNet(nn.Module):
    def __init__(self, model):
        super(CustomResNet, self).__init__()
        self.model = model
        self.part = nn.Sequential(
            model.conv1,
            model.bn1,
            model.relu,
            model.maxpool,
#             model.layer1, # ([4, 64, 64, 64]
#             model.layer2, # ([4, 128, 32, 32]))
#             model.layer3, # ([4, 256, 16, 16])
#             model.layer4 #[4, 512, 8, 8])
        )
      
        self.layer1 = model.layer1
        self.layer2 = model.layer2
        self.layer3 = model.layer3
        self.layer4 = model.layer4

        self.linear = nn.Linear(2048, 28)
        

    def forward(self, x, mixup_hidden = True):
        if mixup_hidden == True:
            layer_mix = random.randint(0,4)
        
            x = self.part(x)
            out = x
            if layer_mix == 0:
                out = mixup_data(out)
                
            out = self.layer1(out)
    
            if layer_mix == 1:
                out = mixup_data(out)
    
            out = self.layer2(out)   
    
            if layer_mix == 2:
                out = mixup_data(out)
    
            out = self.layer3(out)  
    
            if layer_mix == 3:
                out = mixup_data(out)
    
            out = self.layer4(out)        
    
            if layer_mix == 4:
                out = mixup_data(out)
            
            out = F.avg_pool2d(out,out.size(2))
            out = out.view(out.size(0), -1)
            out = self.linear(out)
            
            return out
        
        else:
            x = self.part(x)
            out = x
            out = self.layer1(out)
            out = self.layer2(out)
            out = self.layer3(out)
            out = self.layer4(out)
         
            out = F.avg_pool2d(out,out.size(2))
            out = out.view(out.size(0), -1)
            out = self.linear(out)
            
            return out

    
def getmyresnet50(num_classes=28):
    model_name = 'resnet50'
    model = pretrainedmodels.__dict__[model_name](num_classes=1000, pretrained='imagenet')
    conv1 = model.conv1
    model.conv1 = nn.Conv2d(in_channels=4,
                            out_channels=conv1.out_channels,
                            kernel_size=conv1.kernel_size,
                            stride=conv1.stride,
                            padding=conv1.padding,
                            bias=conv1.bias)
    # copy pretrained weights
    model.conv1.weight.data[:,:3,:,:] = conv1.weight.data
    model.conv1.weight.data[:,3:,:,:] = conv1.weight.data[:,:1,:,:]
    
    Mymodel = CustomResNet(model)
    
    return Mymodel


