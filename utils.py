import os
import sys 
import json
import torch
import shutil
import numpy as np 
from config import config
from torch import nn
import torch.nn.functional as F 
from sklearn.metrics import f1_score
from torch.autograd import Variable
from torch.nn.modules.loss import _WeightedLoss
import torch
import torchvision.transforms as transforms
import math
import random 
from data import HumanDataset
from torchvision import transforms as T
import pandas as pd
from tqdm import tqdm 
from datetime import datetime


imageSize = 512
       
def NonLinR(ImNew):
      theta1 = random.uniform(90,120) #0, 360)
      theta = np.radians(theta1)
      
      c, s = np.cos(theta), np.sin(theta)
      #R = np.array(((c, 0, s), (0, 1, 0), (-s,0,c))) #G
      R = np.array(((1, 0, 0), (0, c, -s), (0,s,c))) #R
      
      D1_,D2_,D3_=[],[],[]
      
      for x in range(imageSize):
          for y in range(imageSize):
              vec = [ImNew[x,y,0],ImNew[x,y,1],ImNew[x,y,2]]
              vctr = np.array(vec) 
              RotMat = abs(R.dot(vctr))
              D1_.append(RotMat[0])
              D2_.append(RotMat[1])
              D3_.append(RotMat[2])
      
      D1 = np.stack(D1_)
      D2 = np.stack(D2_)
      D3 = np.stack(D3_)
      
      images2 = np.zeros(shape=(imageSize*imageSize,3))
      images2[:,0] = D1 
      images2[:,1] = D2 
      images2[:,2] = D3
      
      images2 = np.reshape(images2, (imageSize, imageSize,3))   
      
      images_ = np.zeros(shape=(512,512,4)) 
      images_[:,:,0] = images2[:,:,0]
      images_[:,:,1] = images2[:,:,1]
      images_[:,:,2] = images2[:,:,2]
      images_[:,:,3] = ImNew[:,:,3]
      
       
      images_ = images_.astype(np.uint8)
      
      images_ = augumentor1(images_)
      
      images_ = T.Compose([T.ToPILImage(),T.ToTensor()])(images_)
      return images_

def NonLinG(ImNew):
      theta1 = random.uniform(60, 110)
      theta2 = random.uniform(120, 300)
      theta = random.choice([theta1, theta2])	
      
      theta = np.radians(theta)
      
      c, s = np.cos(theta), np.sin(theta)
      R = np.array(((c, 0, s), (0, 1, 0), (-s,0,c))) #G
      #R = np.array(((1, 0, 0), (0, c, -s), (0,s,c))) #R
      
      D1_,D2_,D3_=[],[],[]
      
      for x in range(imageSize):
          for y in range(imageSize):
              vec = [ImNew[x,y,0],ImNew[x,y,1],ImNew[x,y,2]]
              vctr = np.array(vec) 
              RotMat = abs(R.dot(vctr))
              D1_.append(RotMat[0])
              D2_.append(RotMat[1])
              D3_.append(RotMat[2])
      
      D1 = np.stack(D1_)
      D2 = np.stack(D2_)
      D3 = np.stack(D3_)
      
      images2 = np.zeros(shape=(imageSize*imageSize,3))
      images2[:,0] = D1 
      images2[:,1] = D2 
      images2[:,2] = D3
      
      images2 = np.reshape(images2, (imageSize, imageSize,3))   
      
      images_ = np.zeros(shape=(512,512,4)) 
      images_[:,:,0] = images2[:,:,0]
      images_[:,:,1] = images2[:,:,1]
      images_[:,:,2] = images2[:,:,2]
      images_[:,:,3] = ImNew[:,:,3]
      
       
      images_ = images_.astype(np.uint8)
      
      images_ = augumentor1(images_)
      
      images_ = T.Compose([T.ToPILImage(),T.ToTensor()])(images_)
      return images_
            
def augumentor1(image):   # all these augmentations have been applied in random order 
      flip_aug = iaa.Sequential([
              iaa.OneOf([
                  iaa.Affine(rotate=90),
                  iaa.Affine(rotate=180),
                  iaa.Affine(rotate=270),
                  iaa.Affine(shear=(-16, 16)),
                  iaa.Fliplr(0.5),
                  iaa.Flipud(0.5),
              ])
          ], random_order=True)
    
      crop_aug = iaa.Sometimes(
                  0.5,
                  iaa.Sequential([
                      iaa.OneOf([
                          iaa.CropToFixedSize(288, 288),
                          iaa.CropToFixedSize(320, 320),
                          iaa.CropToFixedSize(352, 352),
                          iaa.CropToFixedSize(384, 384),
                          iaa.CropToFixedSize(416, 416),
                          iaa.CropToFixedSize(448, 448),
                      ])
                  ])
              )
      #pad_aug = iaa.PadToFixedSize(width=512, height=512)

      mul_aug = iaa.Sometimes(0.5, iaa.Multiply((0.5, 1.5), per_channel=0.5)) 
          
      aug = iaa.Sequential([flip_aug, crop_aug, mul_aug])
      image_aug = aug.augment_image(image)   
      
      image_aug =  cv2.resize(image_aug,(config.img_weight,config.img_height))
      
      image_aug = T.Compose([T.ToPILImage(),T.ToTensor()])(image_aug)
        
      return image_aug
    
       
def findInd(label):  # Find Image2 from TRAINING set with same label

    label10 = np.load('LabelInd10.npy')
    label15 = np.load('LabelInd15.npy')
    label27 = np.load('LabelInd27.npy')
    label8 = np.load('LabelInd8.npy')
    label9 = np.load('LabelInd9.npy')
    
    label6 = np.load('LabelInd6.npy')
    label20 = np.load('LabelInd20.npy')
    label12 = np.load('LabelInd12.npy')
    label13 = np.load('LabelInd13.npy')
    label17 = np.load('LabelInd17.npy')
    
    label16 = np.load('LabelInd16.npy')
    label18 = np.load('LabelInd18.npy') 
    label22 = np.load('LabelInd22.npy')
    label24 = np.load('LabelInd24.npy')
    label26 = np.load('LabelInd26.npy') 
    
    
    if label == 10:
        ind2List = label10
    if label == 15:
        ind2List = label15
    if label == 27:
        ind2List = label27
    if label == 8:
        ind2List = label8
    if label == 9:
        ind2List = label9
        
    if label == 6:
        ind2List = label6
    if label == 20:
        ind2List = label20
    if label == 12:
        ind2List = label12
    if label == 13:
        ind2List = label13
    if label == 17:
        ind2List = label17
        
    if label == 16:
        ind2List = label16
    if label == 18:
        ind2List = label18
    if label == 22:
        ind2List = label22
    if label == 24:
        ind2List = label24
    if label == 26:
        ind2List = label26
           
    return ind2List

def per_image_standardization(x):
    y = x.view(-1, x.shape[1]*x.shape[2]*x.shape[3])
    mean = y.mean(dim=1, keepdim = True).expand_as(y)    
    std = y.std(dim=1, keepdim = True).expand_as(y)      
    adjusted_std = torch.max(std, 1.0/torch.sqrt(torch.cuda.FloatTensor([x.shape[1]*x.shape[2]*x.shape[3]])))    
    y = (y- mean)/ adjusted_std
    standarized_input =  y.view(x.shape[0],x.shape[1],x.shape[2],x.shape[3])  
    return standarized_input  
    


class ArcFaceLoss(nn.modules.Module):
    def __init__(self,s=30.0,m=0.5):
        super(ArcFaceLoss, self).__init__()
        self.classify_loss = nn.BCEWithLogitsLoss()#CrossEntropyLoss()
        self.s = s
        self.easy_margin = False
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, logits, labels, epoch=0):
        cosine = logits
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        phi = cosine * self.cos_m - sine * self.sin_m
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        
        total_target_=[]
        bs = labels.size()
        bs = int(bs[0])
        for ini in range(bs):
            TenLabel = labels[ini]
            ReaLabel = LabelValueList[TenLabel]
            target  = np.eye(config.num_classes,dtype=np.float)[ReaLabel].sum(axis=0)
            total_target_.append(target)
        
        total_target = np.stack(total_target_)
        one_hot = torch.from_numpy(np.array(total_target)).float().cuda(non_blocking=True)

               
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s
        loss1 = self.classify_loss(output, one_hot)
        loss2 = self.classify_loss(cosine, one_hot)
        gamma=1
        loss=(loss1+gamma*loss2)/(1+gamma)
        return loss
    
    


def getIndicesList(label):
    Ind2list2 = []
    for ind in trainIndex:
        labelList = list(map(int, all_files.loc[ind].Target.split(' ')))
        if label in labelList:  #check out the samples with same class in train set and make a list of its indices
            Ind2list2.append(ind)
    return Ind2list2
            



def interleave(x, size):
    s = list(x.shape)
    return x.reshape([-1, size] + s[1:]).transpose(0, 1).reshape([-1] + s[1:])


def de_interleave(x, size):
    s = list(x.shape)
    return x.reshape([size, -1] + s[1:]).transpose(0, 1).reshape([-1] + s[1:])

def load_ckp(checkpoint_fpath, model, optimizer):
    checkpoint = torch.load(checkpoint_fpath)
    model.load_state_dict(checkpoint['state_dict'])
    
#     own_state = self.state_dict()
#         for name, param in state_dict.items():
#             if name not in own_state:
#                  continue
#             if isinstance(param, Parameter):
#                 # backwards compatibility for serialized parameters
#                 param = param.data
#             own_state[name].copy_(param)
    
    optimizer.load_state_dict(checkpoint['optimizer'])
    return model, optimizer, checkpoint['epoch'],checkpoint["best_loss"]

def accuracy(prob, truth, threshold=0.5,  is_average=True):
    batch_size = prob.size(0)
    p = prob.detach().view(batch_size,-1)
    t = truth.detach().view(batch_size,-1)

    p = p>threshold
    t = t>0.5
    correct = ( p == t).float()
    accuracy = correct.sum(1)/p.size(1)

    if is_average:
        accuracy = accuracy.sum()/batch_size
        return accuracy
    else:
        return accuracy
    
    
    
class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, smoothing=0.35):
        super(LabelSmoothingCrossEntropy, self).__init__()
        assert smoothing < 1.0
        self.smoothing = smoothing
        self.confidence = 1. - smoothing

    def forward(self, x, target):
        target = target.float() * (self.confidence) + 0.03571 * self.smoothing
        return F.binary_cross_entropy_with_logits(x, target.type_as(x))



        
# evaluate meters
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

# print logger
class Logger(object):
    def __init__(self):
        self.terminal = sys.stdout  #stdout
        self.file = None

    def open(self, file, mode=None):
        if mode is None: mode ='w'
        self.file = open(file, mode)

    def write(self, message, is_terminal=1, is_file=1 ):
        if '\r' in message: is_file=0

        if is_terminal == 1:
            self.terminal.write(message)
            self.terminal.flush()
            #time.sleep(1)

        if is_file == 1:
            self.file.write(message)
            self.file.flush()

    def flush(self):
        # this flush method is needed for python 3 compatibility.
        # this handles the flush command by doing nothing.
        # you might want to specify some extra behavior here.
        pass

# class FocalLoss(nn.Module):
#     def __init__(self, alpha=0.25,gamma=2):
#         super(FocalLoss, self).__init__()
#         self.alpha = alpha
#         self.gamma = gamma

#     def forward(self, x, y):
#         '''Focal loss.
#         Args:
#           x: (tensor) sized [N,D].
#           y: (tensor) sized [N,].
#         Return:
#           (tensor) focal loss.
#         '''
#         t = Variable(y).cuda()  # [N,20]

#         p = x.sigmoid()
#         pt = p*t + (1-p)*(1-t)         # pt = p if t > 0 else 1-p
#         w = self.alpha*t + (1-self.alpha)*(1-t)  # w = alpha if t > 0 else 1-alpha
#         w = w.detach() * (1-pt).pow(self.gamma)
#         return F.binary_cross_entropy_with_logits(x, t, w.detach(), size_average=False)
    
    
class FocalLoss(nn.Module):
    def __init__(self, gamma=2):
        super().__init__()
        self.gamma = gamma

    def forward(self, logit, target, epoch=0):
        target = target.float()
        max_val = (-logit).clamp(min=0)
        loss = logit - logit * target + max_val + \
               ((-max_val).exp() + (-logit - max_val).exp()).log()

        invprobs = F.logsigmoid(-logit * (target * 2.0 - 1.0))
        loss = (invprobs * self.gamma).exp() * loss
        if len(loss.size())==2:
            loss = loss.sum(dim=1)
        return loss.mean()
    
def get_learning_rate(optimizer):
    lr=[]
    for param_group in optimizer.param_groups:
       lr +=[ param_group['lr'] ]

    #assert(len(lr)==1) #we support only one param_group
    lr = lr[0]

    return lr

def time_to_str(t, mode='min'):
    if mode=='min':
        t  = int(t)/60
        hr = t//60
        min = t%60
        return '%2d hr %02d min'%(hr,min)

    elif mode=='sec':
        t   = int(t)
        min = t//60
        sec = t%60
        return '%2d min %02d sec'%(min,sec)


    else:
        raise NotImplementedError



class FocalLoss(nn.Module):
    def __init__(self, gamma=2):
        super().__init__()
        self.gamma = gamma
        

    def forward(self,logits, targets):
        l = logits.reshape(-1)
        t = targets.reshape(-1)
        p = torch.sigmoid(l)
        p = torch.where(t >= 0.5, p, 1-p)
        logp = - torch.log(torch.clamp(p, 1e-4, 1-1e-4))
        loss = logp*((1-p)**self.gamma)
        loss = 28*loss.mean()
        return loss
  

# --------------------------- MULTICLASS LOSSES ---------------------------
class lovasz_soft(nn.Module):
    def __init__(self, classes='present', per_image=False, ignore=None):
        super().__init__()
        self.classes = classes
        self.per_image = per_image
        self.ignore = ignore
        
    def lovasz_grad(self, gt_sorted):
        """
        Computes gradient of the Lovasz extension w.r.t sorted errors
        See Alg. 1 in paper
        """
        p = len(gt_sorted)
        gts = gt_sorted.sum()
        intersection = gts - gt_sorted.float().cumsum(0)
        union = gts + (1 - gt_sorted).float().cumsum(0)
        jaccard = 1. - intersection / union
        if p > 1: # cover 1-pixel case
            jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
    #     print('jaccard',jaccard)
        return jaccard   


    def lovasz_softmax_flat(self, probas, labels, classes='present'):
        """
        Multi-class Lovasz-Softmax loss
          probas: [P, C] Variable, class probabilities at each prediction (between 0 and 1)
          labels: [P] Tensor, ground truth labels (between 0 and C - 1)
          classes: 'all' for all, 'present' for classes present in labels, or a list of classes to average.
        """
        if probas.numel() == 0:
            # only void pixels, the gradients should be 0
            return probas * 0.

        C = probas.size(1)

        label = np.where(labels.cpu() == 1)
        labelss = torch.from_numpy(label[1])

        losses = []
        class_to_sum = list(range(C)) if classes in ['all', 'present'] else classes
        ###############################################################################

        ind = label[0]
        for c in class_to_sum:
            lst = [0] * probas.size()[0]
            fg = torch.tensor(lst)

            fg_ = (labelss == c).float() # foreground for class c


            for kk in range(fg_.size()[0]):
                if fg_[kk] == 1:
                    fg[ind[kk]] = 1
    #         print('fg===>>',fg)
            fg = fg.cuda()     


    #         print('TRYING',fg[])
            if (classes is 'present' and fg.sum() == 0):
                continue
            if C == 1:
                if len(classes) > 1:
                    raise ValueError('Sigmoid output possible only with 1 class')
                class_pred = probas[:, 0]
            else:
                class_pred = probas[:, c]

    #         print('Variable(fg)',Variable(fg))
    #         print('class_pred',class_pred)
            errors = (Variable(fg) - class_pred).abs()
    #         print('errors',errors)
            errors_sorted, perm = torch.sort(errors, 0, descending=True)
    #         print('errors_sorted',errors_sorted)
    #         print('perm',perm)
            perm = perm.data
            fg_sorted = fg[perm]
    #         print('fg_sorted',fg_sorted)
    #         print('*****************************************************************************')
            losses.append(torch.dot(errors_sorted, Variable(self.lovasz_grad(fg_sorted))))
    #     print('losses--->', losses)
        return self.mean(losses)


    def flatten_probas(self,probas, labels, ignore=None):
        """
        Flattens predictions in the batch
        """
    #     print('ignore', ignore)
        if probas.dim() == 3:
            # assumes output of a sigmoid layer
            B, H, W = probas.size()
            probas = probas.view(B, 1, H, W)

        if ignore is None:
            return probas, labels
        valid = (labels != ignore)
        vprobas = probas[valid.nonzero().squeeze()]
        vlabels = labels[valid]
        return vprobas, vlabels

    def xloss(self, logits, labels, ignore=None):
        """
        Cross entropy loss
        """
        return F.cross_entropy(logits, Variable(labels), ignore_index=255)


    def mean(self, l, ignore_nan=False, empty=0):
        """
        nanmean compatible with generators.
        """
        l = iter(l)
        if ignore_nan:
            l = ifilterfalse(isnan, l)
        try:
            n = 1
            acc = next(l)
        except StopIteration:
            if empty == 'raise':
                raise ValueError('Empty mean')
            return empty
        for n, v in enumerate(l, 2):
            acc += v
        if n == 1:
            return acc
        return acc / n
        
#     def lovasz_softmax(probas, labels, classes='present', per_image=False, ignore=None):
    def forward(self, probas, labels): 
        """Multi-class Lovasz-Softmax loss
          probas: [B, C, H, W] Variable, class probabilities at each prediction (between 0 and 1).
                  Interpreted as binary (sigmoid) output with outputs of size [B, H, W].
          labels: [B, H, W] Tensor, ground truth labels (between 0 and C - 1)
          classes: 'all' for all, 'present' for classes present in labels, or a list of classes to average.
          per_image: compute the loss per image instead of per batch
          ignore: void class labels
        """
        if self.per_image:
            loss = self.mean(self.lovasz_softmax_flat(*self.flatten_probas(prob.unsqueeze(0), lab.unsqueeze(0), self.ignore), classes=self.classes)
                              for prob, lab in zip(probas, labels))
        else:
            loss = self.lovasz_softmax_flat(*self.flatten_probas(probas, labels, self.ignore), classes=self.classes)
        return loss
