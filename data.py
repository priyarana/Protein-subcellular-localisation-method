from utils import *
from config import *
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T
from sklearn.preprocessing import MultiLabelBinarizer
from imgaug import augmenters as iaa
import random
import pathlib
import cv2
import torchvision.transforms as transforms
import pandas as pd
from collections import Counter
from itertools import chain
import math
from utils import *
import csv
import random

# set random seed
random.seed(2050)
np.random.seed(2050)
torch.manual_seed(2050)
torch.cuda.manual_seed_all(2050)

TrainRedIms = np.load("path to red channel images",allow_pickle=True)
TrainGreenIms = np.load("path to green channel images",allow_pickle=True)
TrainBlueIms = np.load("path to blue channel images",allow_pickle=True)
TrainYellowIms = np.load("path to yellow channel images",allow_pickle=True)

TestRedIms = np.load("path to red channel images",allow_pickle=True)
TestGreenIms = np.load("path to green channel images",allow_pickle=True)
TestBlueIms = np.load("path to blue channel images",allow_pickle=True)
TestYellowIms = np.load("path to yellow channel images",allow_pickle=True)


# create dataset class
class HumanDataset(Dataset):

    def __init__(self,images_df,augument=True,mode="train"):#,mode1 = "None"):
      
        self.images_df = images_df.copy()
        self.augument = augument
        #self.images_df.Id = self.images_df.Id.apply(lambda x:base_path / x)
        self.mlb = MultiLabelBinarizer(classes = np.arange(0,config.num_classes))
        #self.mlb.fit(np.arange(0,config.num_classes))
        self.mlb_label_train = self.mlb.fit(np.arange(0,config.num_classes))
 
        self.mode = mode
        #self.mode1 = mode1

    def __len__(self):
        return len(self.images_df)

    def __getitem__(self,index):
             
        if not self.mode == "test":  #train
            X = self.Trainread_images(index)        
            labels = np.array(list(map(int, self.images_df.iloc[index].Target.split(' '))))      #always there
            y  = np.eye(config.num_classes,dtype=np.float)[labels].sum(axis=0)                   #<<<<Uncomment for Standard learning
        else:
            X = self.Testread_images(index)
            #y = str(self.images_df.iloc[index].Id.absolute())    
            y = str(self.images_df.iloc[index].Id)  

        if self.augument:
            X = self.augumentor(X)

        X = T.Compose([T.ToPILImage(),T.ToTensor()])(X) 
        return X.float(), y #, index #for semi-super
    
    def getLabel2(self,index):  
        
        labelList = list(map(int, self.images_df.loc[index].Target.split(' ')))
        y  = np.eye(config.num_classes,dtype=np.float)[labelList].sum(axis=0)                   #<<<<Uncomment for Standard learning
        
        return y
        
    def Trainread_images1(self,index):
        
        row = self.images_df.loc[index]
        #filename = str(row.Id.absolute())
        filename = str(row.Id)

        if config.channels == 4:
            images = np.zeros(shape=(512,512,4))
        else: 
            images = np.zeros(shape=(512,512,3))              #use only rgb channels

        namelistR= TrainRedIms[:,0]
        namelistG= TrainGreenIms[:,0]
        namelistB= TrainBlueIms[:,0]
        namelistY= TrainYellowIms[:,0]
        
        r_ = np.where(namelistR == filename)
        r = r_[0]
        r = r[0]
        r = TrainRedIms[r,1]
        
        g_ = np.where(namelistG == filename)
        g = g_[0]
        g = g[0]
        g = TrainGreenIms[g,1]
        
        b_ = np.where(namelistB == filename)
        b = b_[0]
        b = b[0]
        b = TrainBlueIms[b,1]
        
        y_ = np.where(namelistY == filename)
        y = y_[0]
        y = y[0]
        y = TrainYellowIms[y,1]

        images[:,:,0] = r.astype(np.uint8) 
        #images[:,:,0] = y.astype(np.uint8)
        images[:,:,1] = g.astype(np.uint8)
        images[:,:,2] = y.astype(np.uint8)

        if config.channels == 4:
            images[:,:,3] = b.astype(np.uint8)
        images = images.astype(np.uint8)
        
        if config.img_height == 512:
            return images
        else:
            return cv2.resize(images,(config.img_weight,config.img_height))
            
    def Trainread_images(self,index):
        
        row = self.images_df.iloc[index]
        #filename = str(row.Id.absolute())
        filename = str(row.Id)

        if config.channels == 4:
            images = np.zeros(shape=(512,512,4))
        else: 
            images = np.zeros(shape=(512,512,3))              #use only rgb channels

        namelistR= TrainRedIms[:,0]
        namelistG= TrainGreenIms[:,0]
        namelistB= TrainBlueIms[:,0]
        namelistY= TrainYellowIms[:,0]
        
        r_ = np.where(namelistR == filename)
        r = r_[0]
        r = r[0]
        r = TrainRedIms[r,1]
        
        g_ = np.where(namelistG == filename)
        g = g_[0]
        g = g[0]
        g = TrainGreenIms[g,1]
        
        b_ = np.where(namelistB == filename)
        b = b_[0]
        b = b[0]
        b = TrainBlueIms[b,1]
        
        y_ = np.where(namelistY == filename)
        y = y_[0]
        y = y[0]
        y = TrainYellowIms[y,1]

        images[:,:,0] = r.astype(np.uint8) 
        #images[:,:,0] = y.astype(np.uint8)
        images[:,:,1] = g.astype(np.uint8)
        images[:,:,2] = y.astype(np.uint8)

        if config.channels == 4:
            images[:,:,3] = b.astype(np.uint8)
        images = images.astype(np.uint8)
        
        if config.img_height == 512:
            return images
        else:
            return cv2.resize(images,(config.img_weight,config.img_height))
            
    def Testread_images(self,index):
        
        row = self.images_df.iloc[index]
        #filename = str(row.Id.absolute())
        filename = str(row.Id)

        if config.channels == 4:
            images = np.zeros(shape=(512,512,4))
        else: 
            images = np.zeros(shape=(512,512,3))              #use only rgb channels

        namelistR= TestRedIms[:,0]
        namelistG= TestGreenIms[:,0]
        namelistB= TestBlueIms[:,0]
        namelistY= TestYellowIms[:,0]
        
        r_ = np.where(namelistR == filename)
        r = r_[0]
        r = r[0]
        r = TestRedIms[r,1]
        
        g_ = np.where(namelistG == filename)
        g = g_[0]
        g = g[0]
        g = TestGreenIms[g,1]
        
        b_ = np.where(namelistB == filename)
        b = b_[0]
        b = b[0]
        b = TestBlueIms[b,1]
        
        y_ = np.where(namelistY == filename)
        y = y_[0]
        y = y[0]
        y = TestYellowIms[y,1]

        images[:,:,0] = r.astype(np.uint8) 
        #images[:,:,0] = y.astype(np.uint8)
        images[:,:,1] = g.astype(np.uint8)
        images[:,:,2] = y.astype(np.uint8)

        if config.channels == 4:
            images[:,:,3] = b.astype(np.uint8)
        images = images.astype(np.uint8)
        
        if config.img_height == 512:
            return images
        else:
            return cv2.resize(images,(config.img_weight,config.img_height))

    def augumentor(self,image):   # all these augmentations have been applied in random order 
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
        pad_aug = iaa.PadToFixedSize(width=512, height=512)

        mul_aug = iaa.Sometimes(0.5, iaa.Multiply((0.5, 1.5), per_channel=0.5)) 
        
#       sup_aug = iaa.Sometimes(0.5, iaa.Superpixels(p_replace=(0, 1.0), n_segments=(20, 200)))
#       blur_aug = iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255), per_channel=0.5)   #iaa.Sometimes(0.5, iaa.GaussianBlur((0.0, 1.0)))      
#       contrast_aug =  iaa.LinearContrast((0.5, 2.0), per_channel=0.5)
            
        aug = iaa.Sequential([flip_aug, crop_aug, pad_aug, mul_aug])
        image_aug = aug.augment_image(image)     
        
        size= image_aug.shape
        if size == (512,512,4):
            return image_aug
        else:
            return cv2.resize(image_aug,(config.img_weight,config.img_height))        
        return image_aug
    
    
    def WeakAugumentor(self,image):   # all these augmentations have been applied in random order
        image = T.Compose([T.ToPILImage()])(image)
        weak_im = transforms.RandomHorizontalFlip(p=0.75)(image)
        resized_im = transforms.Resize(512)(weak_im)
        translated = transforms.RandomCrop(size=512, 
                                         padding=int(512*0.125), 
                                         padding_mode='reflect')(resized_im)
        translated = np.array(translated)
        return translated
        
#         flipAug = iaa.Fliplr(0.5)         
#         shearAug = iaa.Affine(shear=(-12.5, 12.5))
        
#         aug = iaa.Sequential([flip_aug, shearAug])
#         image_aug = aug.augment_image(image)     
  
    def tta_aug(self,image):   # all these augmentations have been applied in random order
        
        image = T.Compose([T.ToPILImage()])(image)
        weak_im = transforms.RandomHorizontalFlip(p=0.5)(image)
        translated = transforms.RandomVerticalFlip(p=0.5)(weak_im)
        translated = np.array(translated)
        return translated
    
    

    def augumentor1(self,image):   # all these augmentations have been applied in random order
        #augment_img =iaa.Sometimes(0.85, iaa.Sequential([
        augment_img =iaa.Sequential([
                    iaa.OneOf([
                          iaa.Affine(rotate=90),
                          iaa.Affine(rotate=180),
                          iaa.Affine(rotate=270),
                          iaa.Affine(shear=(-16, 16)),
                          iaa.Fliplr(0.5),
                          iaa.Flipud(0.5),
                          iaa.Sometimes(0.5, iaa.Multiply((0.5, 1.5), per_channel=0.5)),
                          iaa.Sometimes(0.5, iaa.Superpixels(p_replace=(0, 1.0), n_segments=(20, 200)))
                    ])
                ], random_order=True)

        image_aug = augment_img.augment_image(image)  #standard Aug ends here
        return image_aug
    
#         augment_img = iaa.Sequential([
#                 iaa.OneOf([
#                     iaa.Fliplr(0.5),
#                     iaa.Affine(translate_percent={"x": (-0.125, 0.125), "y": (-0.125, 0.125)}),
#                 ])
#             ], random_order=True)    
#         image_aug = augment_img.augment_image(image)  #standard Aug ends here
#         return image_aug
#         PIL_image = Image.fromarray(image_aug) 
#         policy = ImageNetPolicy()
#         transformed = policy(PIL_image)
#         trns_imge = np.array(transformed)    
#         return trns_imge
        
def get_index_dic():
    """ build a dict with class as key and img_ids as values
    :return: dict()
    """
    num_classes = 28 #len(self.get_ann_info(0)['labels'])
    gt_labels = []
    idx2img_id = []

    img_id2idx = dict()

    condition_prob = np.zeros([num_classes, num_classes])

    index_dic = [[] for i in range(num_classes)]
    co_labels = [[] for _ in range(num_classes)]

    all_files = pd.read_csv("D:/DATAs/Kaggle512/train.csv")
    trainIndex = np.load('Index/train1.npy')  
    train_data_list = all_files.loc[trainIndex]
    
    Ran = len(train_data_list)
    print(Ran)
    #for i, img_info in enumerate(self.img_infos):
    for i in range(Ran):#31072):
        img_id = train_data_list.iloc[i].Id# train_data_list.Id[i]

        labels2 = np.array(list(map(int, train_data_list.iloc[i].Target.split(' '))))
        label  = np.eye(28,dtype=np.float)[labels2].sum(axis=0)    
        
        gt_labels.append(label)
        idx2img_id.append(img_id)
        img_id2idx[img_id] = i

        la = np.where(np.asarray(label) == 1)[0]
        for idx in la:
            index_dic[idx].append(i)
            co_labels[idx].append(idx)

    for cla in range(num_classes):
        cls_labels = co_labels[cla]
        num = len(cls_labels)
        condition_prob[cla] = np.sum(np.asarray(cls_labels), axis=0) / num
    ''' save original dataset statistics, run once!'''
#         if self.save_info:
#             self._save_info(gt_labels, img_id2idx, idx2img_id, condition_prob)

    return index_dic, co_labels
#     else:
#         return index_dic
        


def create_class_weight(labels_dict, mu=0.5):
    total = sum(labels_dict.values())
    keys = labels_dict.keys()
    class_weight = dict()
    class_weight_log = dict()
    for key in keys:
        score = float(total) / float(labels_dict[key])
        score_log = math.log(mu * total / float(labels_dict[key]))
        class_weight[key] = round(score, 2) if score > config.min_sampling_limit else round(config.min_sampling_limit, 2)
        class_weight_log[key] = round(score_log, 2) if score_log > config.min_sampling_limit else round(config.min_sampling_limit, 2)
    return class_weight, class_weight_log


#def process_df(train_df, external_df, test_df):
def process_df(train_df):
    train_df['target_vec'] = train_df['Target'].map(lambda x: list(map(int, x.strip().split())))
    #external_df['target_vec'] = external_df['Target'].map(lambda x: list(map(int, x.strip().split())))
    train_df['is_external'] = 0
    #external_df['is_external'] = 1
    #test_df['is_external'] = 0

    all_df = pd.concat([train_df], ignore_index=True)
    if config.is_train:
        count_labels = Counter(list(chain.from_iterable(all_df['target_vec'].values)))
        class_weight, class_weight_log = create_class_weight(count_labels, 0.3)
        cwl = np.ones((len(class_weight_log)))
        for key, value in class_weight_log.items():
            cwl[key] = value

        def calculate_freq(row):
            row.loc['freq'] = 0
            for num in row.target_vec:
                row.loc['freq'] = max(row.loc['freq'], class_weight_log[num])
            return row

        all_df = all_df.apply(calculate_freq, axis=1)
        # print(count_labels)
        # print(class_weight_log)
        return all_df, cwl #test_df, cwl
    else:
        all_df['freq'] = 1
        cwl = np.ones(config.num_classes)
        return all_df, cwl #test_df, cwl

