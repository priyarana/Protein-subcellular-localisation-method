import os 
import time 
import json 
import torch 
import random 
import warnings
import torchvision
import numpy as np 
import pandas as pd 
from imgaug import augmenters as iaa
import pathlib
from utils import *
from data import HumanDataset
from tqdm import tqdm 
from config import config
from datetime import datetime
from torch import nn,optim
from collections import OrderedDict
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
from sklearn.model_selection import train_test_split
from timeit import default_timer as timer
from sklearn.metrics import f1_score
import torch.nn.functional as F
from torch.utils.data.sampler import WeightedRandomSampler
import operator
import cv2

from models.model import*
from data import process_df


import gc 
gc.collect() 
torch.cuda.empty_cache()

# 1. set random seed
random.seed(2050)
np.random.seed(2050)
torch.manual_seed(2050)
torch.cuda.manual_seed_all(2050)

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
torch.backends.cudnn.benchmark = True
warnings.filterwarnings('ignore')

if not os.path.exists("./logs/"):
    os.mkdir("./logs/")

log = Logger()
log.open("logs/%s_log_train.txt"%config.model_name,mode="a")
log.write("\n----------------------------------------------- [START %s] %s\n\n" % (datetime.now().strftime('%Y-%m-%d %H:%M:%S'), '-' * 51))
log.write('                           |------------ Train -------------|----------- Valid -------------|----------Best Results---------|------------|\n')
log.write('mode     iter     epoch    |         loss   f1_macro        |         loss   f1_macro       |         loss   f1_macro       | time       |\n')
log.write('-------------------------------------------------------------------------------------------------------------------------------\n')


 


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

    label10 = np.load('/g/data/nk53/pr2894/PSL/Code/1/LabelInd10.npy')
    label15 = np.load('/g/data/nk53/pr2894/PSL/Code/1/LabelInd15.npy')
    label27 = np.load('/g/data/nk53/pr2894/PSL/Code/1/LabelInd27.npy')
    label8 = np.load('/g/data/nk53/pr2894/PSL/Code/1/LabelInd8.npy')
    label9 = np.load('/g/data/nk53/pr2894/PSL/Code/1/LabelInd9.npy')
    
    label6 = np.load('/g/data/nk53/pr2894/PSL/Code/1/LabelInd6.npy')
    label20 = np.load('/g/data/nk53/pr2894/PSL/Code/1/LabelInd20.npy')
    label12 = np.load('/g/data/nk53/pr2894/PSL/Code/1/LabelInd12.npy')
    label13 = np.load('/g/data/nk53/pr2894/PSL/Code/1/LabelInd13.npy')
    label17 = np.load('/g/data/nk53/pr2894/PSL/Code/1/LabelInd17.npy')
    
    label16 = np.load('/g/data/nk53/pr2894/PSL/Code/1/LabelInd16.npy')
    label18 = np.load('/g/data/nk53/pr2894/PSL/Code/1/LabelInd18.npy') 
    label22 = np.load('/g/data/nk53/pr2894/PSL/Code/1/LabelInd22.npy')
    label24 = np.load('/g/data/nk53/pr2894/PSL/Code/1/LabelInd24.npy')
    label26 = np.load('/g/data/nk53/pr2894/PSL/Code/1/LabelInd26.npy') 
    
    
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

def MultiMixUp(Ind1,Ind2,train_gen):
    Im1 = train_gen.Trainread_images1(Ind1)     
    Im2 = train_gen.Trainread_images1(Ind2)  
    
    random.seed(datetime.now())
    
    alpha = np.random.uniform(low=0.35, high=0.65, size=(imageSize,imageSize,4)) #np.random.random((48,48,3))    # 

    ImNew = alpha * Im1 + (1-alpha) * Im2   #Interpolation   
    
    return ImNew
    
    
def MediumComp(LlabListMed,train_gen):
    ct = -1
    for label in LlabListMed:
        #print('main label',label)
        ct = ct+1
        
        Indicelist = findInd(label)
        random.seed(datetime.now())
        Im1Ind = random.choice(Indicelist)
        Im2Ind = random.choice(Indicelist)
        Im3Ind = random.choice(Indicelist)
        
        Im1 = train_gen.Trainread_images1(Im1Ind) 
        Im2 = train_gen.Trainread_images1(Im2Ind) 
        Im3 = train_gen.Trainread_images1(Im3Ind) 
        
        Im2 = augumentor1(Im2)
        Im1 = augumentor1(Im1)
        Im3 = augumentor1(Im3)
        
        labIm1 = train_gen.getLabel2(Im1Ind)
        labIm1_ = torch.from_numpy(np.array(labIm1)).float().cuda(non_blocking=True)
        labIm1_ = labIm1_[None, :]
        
        labIm2 = train_gen.getLabel2(Im2Ind)
        labIm2_ = torch.from_numpy(np.array(labIm2)).float().cuda(non_blocking=True)
        labIm2_ = labIm2_[None, :]
        
        labIm3 = train_gen.getLabel2(Im3Ind)
        labIm3_ = torch.from_numpy(np.array(labIm3)).float().cuda(non_blocking=True)
        labIm3_ = labIm3_[None, :]

            
        Im1 = T.Compose([T.ToPILImage(),T.ToTensor()])(Im1) 
        Im2 = T.Compose([T.ToPILImage(),T.ToTensor()])(Im2) 
        Im3 = T.Compose([T.ToPILImage(),T.ToTensor()])(Im3) 
        
        Im1 = Im1.cuda(non_blocking=True)
        Im1 = Im1[None, :, :]
        Im2 = Im2.cuda(non_blocking=True)
        Im2 = Im2[None, :, :]
        Im3 = Im3.cuda(non_blocking=True)
        Im3 = Im3[None, :, :]
            
        if ct==0:
            Min_input =  torch.cat((Im1, Im2, Im3),0).cuda(non_blocking=True)
            Min_target = torch.cat((labIm1_, labIm2_, labIm3_),0)
        else:
            Min_input = torch.cat((Min_input, Im1, Im2, Im3),0).cuda(non_blocking=True) 
            Min_target = torch.cat((Min_target, labIm1_, labIm2_, labIm3_),0)
            
    return Min_input,Min_target

def train(train_loader,model,criterion,optimizer,epoch,valid_loss,best_results,start,train_gen):
    losses = AverageMeter()
    f1 = AverageMeter()
    model.train()
    
    for i,(images,target) in enumerate(train_loader):
        images = images.cuda(non_blocking=True)
#         print(type(target))
#         print(target)
        target = torch.from_numpy(np.array(target)).float().cuda(non_blocking=True)
        output = model(images)
        
        random.seed(datetime.now())
        kt2 = random.randint(0, 3)   #gives values = 0,1,2,3
        
        if kt2 == 0 or kt2 == 1:
            LlabList = [8,9,10,15,27]
            ct = -1
            for label in LlabList:
                ct = ct+1
                
                Indicelist = findInd(label)
                random.seed(datetime.now())
                Im1Ind = random.choice(Indicelist)
                Im2Ind = random.choice(Indicelist)
                
                Im1 = train_gen.Trainread_images1(Im1Ind) 
                Im2 = train_gen.Trainread_images1(Im2Ind) 
                Im2 = augumentor1(Im2)

                
                WeakAug = MultiMixUp(Im1Ind,Im2Ind,train_gen)  
               
                kt = random.randint(0, 1) #

                if kt == 0:

                    
                    WeakAug = NonLinG(WeakAug)
                    Weaks = WeakAug.cuda(non_blocking=True)
                    Weaks = Weaks[None, :, :]                            
                else:
                    WeakAug = NonLinR(WeakAug)
                    Weaks = WeakAug.cuda(non_blocking=True)
                    Weaks = Weaks[None, :, :]                            
                
                Im1 = augumentor1(Im1)
                
                labIm1 = train_gen.getLabel2(Im1Ind)
                labIm1_ = torch.from_numpy(np.array(labIm1)).float().cuda(non_blocking=True)
                labIm1_ = labIm1_[None, :]
                
                labIm2 = train_gen.getLabel2(Im2Ind)
                labIm2_ = torch.from_numpy(np.array(labIm2)).float().cuda(non_blocking=True)
                labIm2_ = labIm2_[None, :]
    
                MixLabel = np.add(labIm1, labIm2)  
                MixLabel = np.array([1.0 if val>1 else 0.0 for val in MixLabel])
                MixLabel = torch.from_numpy(np.array(MixLabel)).float().cuda(non_blocking=True)
                MixLabel = MixLabel[None, :]
                    
                Im1 = T.Compose([T.ToPILImage(),T.ToTensor()])(Im1) 
                Im2 = T.Compose([T.ToPILImage(),T.ToTensor()])(Im2) 
                
                Im1 = Im1.cuda(non_blocking=True)
                Im1 = Im1[None, :, :]
                Im2 = Im2.cuda(non_blocking=True)
                Im2 = Im2[None, :, :]
                    
                if ct==0:
                    Min_input =  torch.cat((Im1, Im2, Weaks),0).cuda(non_blocking=True)
                    Min_target = torch.cat((labIm1_, labIm2_, MixLabel),0)
                else:
                    Min_input = torch.cat((Min_input, Im1, Im2, Weaks),0).cuda(non_blocking=True) 
                    Min_target = torch.cat((Min_target, labIm1_, labIm2_, MixLabel),0)
        elif kt2 == 2:
          LlabListMed = [20,17,24,26,16]
          Min_input,Min_target = MediumComp(LlabListMed,train_gen)
        else:
          LlabListMed = [13,12,22,18,6]
          Min_input,Min_target = MediumComp(LlabListMed,train_gen)

        total_input =  torch.cat((images,Min_input))
        total_target = torch.cat((target,Min_target))

        logits = model(total_input)          
        loss = criterion(logits,total_target)  
        losses.update(loss.item(),images.size(0))

        f1_batch = np.array([0.0])
        f1.update(f1_batch.item(),images.size(0))
        
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        print('\r',end='',flush=True)
        message = '%s %5.1f %6.1f         |         %0.3f  %0.3f           |         %0.3f  %0.3f         |         %s  %s    | %s' % (\
                "train", i/len(train_loader) + epoch, epoch,
                losses.avg, f1.avg, 
                valid_loss[0], valid_loss[1], 
                str(best_results[0])[:8],str(best_results[1])[:8],
                time_to_str((timer() - start),'min'))
        print(message , end='',flush=True)
    log.write("\n")
    
    return [losses.avg,f1.avg]

# 2. evaluate fuunction
def evaluate(val_loader,model,criterion,epoch,train_loss,best_results,start):

    losses = AverageMeter()
    f1 = AverageMeter()
    model.cuda()
    model.eval()

    with torch.no_grad():
        for i, (images,target) in enumerate(val_loader):
            images_var = images.cuda(non_blocking=True)
            target = torch.from_numpy(np.array(target)).float().cuda(non_blocking=True)

            output = model(images_var)
            #output = output[0]
            
            if i==0:
                total_output = output
                total_target = target
            else:
                total_output = torch.cat([total_output, output],0)
                total_target = torch.cat([total_target, target],0)
        
        loss = criterion(total_output, total_target)
        losses.update(loss.item(),images_var.size(0))
        
        
        f1List=[]
        f1ListAll = []
        
        cc = total_output.sigmoid().cpu().data.numpy()
        
        #np.save('Res20',cc)         #--------------------------------->>
        
        
        f1_ = f1_score(total_target.cpu(), total_output.sigmoid().cpu().data.numpy() > 0.50, average='macro')
        f1_All = f1_score(total_target.cpu(), total_output.sigmoid().cpu().data.numpy() > 0.50,  average=None)
        
        f1List.append(f1_)
        f1ListAll.append(f1_All)
            
        index_, f1_batch = max(enumerate(f1List), key=operator.itemgetter(1))  
        print('f1_batch',f1_batch)
        print('f1_listAll',f1ListAll[index_])

        
        f1.update(f1_batch,images_var.size(0))
        print('\r',end='',flush=True)
        message = '%s   %5.1f %6.1f         |         %0.3f           |         %0.3f  %0.4f         |         %s  %s    | %s' % (\
                    "val", i/len(val_loader) + epoch, epoch,                    
                    train_loss[0], 
                    losses.avg, f1.avg,
                    str(best_results[0])[:8],str(best_results[1])[:8],
                    time_to_str((timer() - start),'min'))
        
        print(message, end='',flush=True)
        log.write("\n")
        
    return [losses.avg,f1.avg]


# 3. test model on public dataset and save the probability matrix
def test(test_loader,model,criterion,epoch,flag):

    losses = AverageMeter()
    f1 = AverageMeter()
    # switch mode for evaluation
    model.cuda()
    model.eval()

    with torch.no_grad():
        for i, (images,target) in enumerate(test_loader):
            images_var = images.cuda(non_blocking=True)
            target = torch.from_numpy(np.array(target)).float().cuda(non_blocking=True)

            output = model(images_var)
            #output = output[0]
            
            if i==0:
                total_output = output
                total_target = target
            else:
                total_output = torch.cat([total_output, output],0)
                total_target = torch.cat([total_target, target],0)
        
        loss = criterion(total_output, total_target)
        losses.update(loss.item(),images_var.size(0))
        
        
        f1List=[]
        f1ListAll = []
        
        cc = total_output.sigmoid().cpu().data.numpy()
        
        f1_ = f1_score(total_target.cpu(), total_output.sigmoid().cpu().data.numpy() > 0.50, average='macro')
        f1_All = f1_score(total_target.cpu(), total_output.sigmoid().cpu().data.numpy() > 0.50,  average=None)
        
        f1List.append(f1_)
        f1ListAll.append(f1_All)
            
        index_, f1_batch = max(enumerate(f1List), key=operator.itemgetter(1))  
        print('f1_batch of Test set',f1_batch)
        print('f1_listAll of Test set',f1ListAll[index_])

     
def save_checkpoint(state, is_best_loss,is_best_f1,fold,model,test_loader,criterion):
    #filename = config.weights + config.model_name + os.sep +str(fold) + os.sep + "checkpoint.pth.tar"
    filename = config.weights + config.model_name + os.sep + "checkpoint.pth.tar"
    torch.save(state, filename)
    if is_best_loss:
        shutil.copyfile(filename,"%s/%s_fold_%s_model_best_loss.pth.tar"%(config.best_models,config.model_name,str(fold)))
        print("SAVED--Loss---------->")
        best_model = torch.load("%s/%s_fold_%s_model_best_loss.pth.tar"%(config.best_models,config.model_name,str(fold)))
        model.load_state_dict(best_model["state_dict"])
        test(test_loader,model,criterion,fold,'loss')
    if is_best_f1:
        shutil.copyfile(filename,"%s/%s_fold_%s_model_best_f1.pth.tar"%(config.best_models,config.model_name,str(fold)))
        print("SAVED--F1---------->")
        best_model = torch.load("%s/%s_fold_%s_model_best_f1.pth.tar"%(config.best_models,config.model_name,str(fold)))
        model.load_state_dict(best_model["state_dict"])
        test(test_loader,model,criterion,fold,'f1')
        
# 4. main function
    
def main():
    fold = 0
    # 4.1 mkdirs
    if not os.path.exists(config.submit):
        os.makedirs(config.submit)
    if not os.path.exists(config.weights + config.model_name + os.sep +str(fold)):
        os.makedirs(config.weights + config.model_name + os.sep +str(fold))
    if not os.path.exists(config.best_models):
        os.mkdir(config.best_models)
    if not os.path.exists("./logs/"):
        os.mkdir("./logs/")
    
    # 4.2 get model    
    model = get_resnet50()  #highest
    model.cuda() 

    # criterion
    optimizer = optim.SGD(model.parameters(),lr = config.lr,momentum=0.9,weight_decay=1e-4)
    #optimizer = optim.Adam(model.parameters(),lr = 0.0005,betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)  
    
    criterion = nn.BCEWithLogitsLoss().cuda()


    start_epoch = 0
    best_loss = 999
    best_f1 = 0
    best_results = [np.inf,0]
    val_metrics = [np.inf,0]
    resume = False
    
    all_files = pd.read_csv("path to train.csv")
    
    trainIndex_ = np.load("path to the list of indices of training images ")     
    testIndex = np.load("path to the list of indices of test images (TS1)")
    
    train_data_list_ = all_files.loc[trainIndex_]
    test_files = all_files.loc[testIndex]
    
    trainIndex = np.load('path to the list of indices of training images for fold 1/2/3/4/5')    
    valIndex = np.load('path to the list of indices of validation images for fold 1/2/3/4/5 ')
        
    train_data_list = train_data_list_.iloc[trainIndex]
    val_data_list = train_data_list_.iloc[valIndex]
    
    print(train_data_list.shape)
    print(val_data_list.shape)
    print(test_files.shape)
    
    MinLabs = ['8','9','10','15','27','20','17','24','26','13','16','12','22','18','6']
    
    for minLb in MinLabs:
        print('minLb',minLb)
        LabelInd_ = []
        for ind in train_data_list.index:
            labe = train_data_list.Target.loc[ind]
            labe = labe.split() 
            if minLb in labe:
                LabelInd_.append(ind)
        LabelInd = np.stack(LabelInd_)
        np.save('/LabelInd'+ minLb,LabelInd)
        print('shape',LabelInd.shape)

    train_gen = HumanDataset(train_data_list,augument=True,mode="train") 
    train_loader = DataLoader(train_gen,batch_size=config.batch_size,drop_last=True,pin_memory=True,num_workers=4)

    val_gen = HumanDataset(val_data_list,augument=False,mode="train")
    val_loader = DataLoader(val_gen,batch_size=config.batch_size,shuffle=False,pin_memory=True,num_workers=4)

    test_gen = HumanDataset(test_files,augument=False,mode="train")
    test_loader = DataLoader(test_gen,1,shuffle=False,pin_memory=True,num_workers=4)

    scheduler = lr_scheduler.StepLR(optimizer,step_size=15,gamma=0.05)
    #scheduler = lr_scheduler.StepLR(optimizer,step_size=30,gamma=0.1)
    
    ckp_path = "/g/data/nk53/pr2894/PSL/Code/1/checkpoints/best_models/bninception_bcelog_fold_33_model_best_f1.pth.tar"
    model, optimizer, start_epoch, load_loss = load_ckp(ckp_path, model, optimizer)
    best_results = [load_loss,0.713393]
    print('load_loss', load_loss)
    start_epoch = 35
    
    start = timer()

    
    #train
    for epoch in range(start_epoch,config.epochs):
        
        scheduler.step(epoch)
        # train
        lr = get_learning_rate(optimizer)
        print('lr--->',lr)
        train_metrics = train(train_loader,model,criterion,optimizer,epoch,val_metrics,best_results,start,train_gen)
        # val
        val_metrics = evaluate(val_loader,model,criterion,epoch,train_metrics,best_results,start)
         
        # check results 
        is_best_loss = val_metrics[0] < best_results[0]
        best_results[0] = min(val_metrics[0],best_results[0])
        is_best_f1 = val_metrics[1] > best_results[1]
        best_results[1] = max(val_metrics[1],best_results[1])   
        # save model
        save_checkpoint({
                    "epoch":epoch + 1,
                    "model_name":config.model_name,
                    "state_dict":model.state_dict(),
                    "best_loss":best_results[0],
                    "optimizer":optimizer.state_dict(),
                    "fold":fold,
                    "best_f1":best_results[1],
        },is_best_loss,is_best_f1,epoch,model,test_loader,criterion) 
        # print logs
        print('\r',end='',flush=True)
        log.write('%s  %5.1f %6.1f         |         %0.3f  %0.3f           |         %0.3f  %0.4f         |         %s  %s    | %s' % (\
                "best", epoch, epoch,                    
                train_metrics[0], train_metrics[1], 
                val_metrics[0], val_metrics[1],
                str(best_results[0])[:8],str(best_results[1])[:8],
                time_to_str((timer() - start),'min'))
            )
        log.write("\n")
        time.sleep(0.01)
        
    
if __name__ == "__main__":
    main()
