# coding:utf-8
from torch.utils.data import DataLoader,Dataset
import os
from config import cfg
from skimage import io, transform
import random
import numpy as np
from imgaug import augmenters as iaa
import imgaug as ia
from PIL import Image
import matplotlib.pyplot as plt
import torch


class DataSet(Dataset):
    def __init__(self,pharse,root = './data',cfg = cfg):
        super(Dataset,self).__init__()
        self.pharse = pharse
        self.img_path = root + r'/imgs'
        self.lable_path = root + r'/masks'
        self.files = self.load_file()
        # self.files = [self.files[i] for i in range(16)]
        self.cfg = cfg
        random.shuffle(self.files)

    def __getitem__(self, index):
        img = io.imread(os.path.join(self.img_path,self.files[index]))
        mask = io.imread(os.path.join(self.lable_path,self.files[index]))
        if self.pharse =='train':
            img,mask = self.aug(img,mask)
        img = np.array(img)
        mask = np.array(mask)
        mask = mask[:,:,np.newaxis]
        img = self.processing(img)
        mask = np.transpose(mask,(2,0,1))
        img = np.transpose(img,(2,0,1))

        # print(self.files[index])
        # self.visulization_from_array(img,mask)
        return img,mask

    def visulization_from_array(self,img,mask):
        img = np.transpose(img,(1,2,0))
        mask = np.transpose(mask,(1,2,0))
        # R,G,B = mask.copy(),mask.copy(),mask.copy()
        B = mask.copy()   # 蓝色通道
        B[B == 1] = 0
        B[B == 2] = 0
        B[B == 3] = 255
        B[B == 0] = 0

        G = mask.copy()   # 绿色通道
        G[G == 1] = 0
        G[G == 2] = 255
        G[G == 3] = 0
        G[G == 0] = 0

        R = mask.copy()   # 红色通道
        R[R == 1] = 255
        R[R == 2] = 0
        R[R == 3] = 0
        R[R == 0] = 0

        vis_mask = np.dstack((R,G,B))
        # print(vis_mask)
        vis_mask = Image.fromarray(vis_mask)
        vis_img = Image.fromarray(img)
        plt.imshow(vis_img)
        plt.show()
        plt.imshow(vis_mask)
        plt.show()

    def aug(self,img,mask):
        flipper = iaa.Fliplr(0.5).to_deterministic()
        mask = flipper.augment_image(mask)
        img = flipper.augment_image(img)
        vflipper = iaa.Flipud(0.5).to_deterministic()
        img = vflipper.augment_image(img)
        mask = vflipper.augment_image(mask)
        if random.random() < 0.5:
            rot_time = random.choice([1, 2, 3])
            for i in range(rot_time):
                img = np.rot90(img)
                mask = np.rot90(mask)
        return img, mask

    def processing(self,img):
        # print(img.shape)
        img = img / 255
        img[:, :,0] = img[:,:,0] - self.cfg.DATA_MEAN[0]
        img[:, :,1] = img[:, :,1] - self.cfg.DATA_MEAN[1]
        img[:, :,2] = img[:, :,2] - self.cfg.DATA_MEAN[2]

        img[:, :,0] = img[:, :,0] / self.cfg.DATA_STD[0]
        img[:, :,1] = img[:, :,1] / self.cfg.DATA_STD[1]
        img[:, :,2] = img[:, :,2] / self.cfg.DATA_STD[2]

        return img

    def load_file(self):
        names = None
        with open('./data/{}.txt'.format(self.pharse))as f:
            names = f.readlines()

        names = [name.strip() for name in names]
        return names

    def __len__(self):
        return len(self.files)

class Test_DataSet(Dataset):
    def __init__(self,cfg,idx,root = './data'):
        super(Dataset,self).__init__()
        self.img_path = root + r'/test/' + str(idx)
        self.imgs = os.listdir(self.img_path)
        self.cfg = cfg
        # self.test_aug = iaa.Rot90(k=1)

    def load_file(self):
        files = os.listdir(self.img_path)
        imgs =[]
        for file in files:
            sub_file = os.path.join(self.img_path,file)
            sub_imgs = os.listdir(sub_file)
            imgs.append(sub_imgs)
        return imgs

    def __getitem__(self, index):
        img_name = self.imgs[index]
        img = io.imread(os.path.join(self.img_path,img_name))
        img = np.array(img)
        img = self.processing(img)
        img = np.transpose(img,(2,0,1))
        return img,img_name

    def processing(self,img):
        img = img / 255
        img[:, :,0] = img[:,:,0] - self.cfg.DATA_MEAN[0]
        img[:, :,1] = img[:, :,1] - self.cfg.DATA_MEAN[1]
        img[:, :,2] = img[:, :,2] - self.cfg.DATA_MEAN[2]

        img[:, :,0] = img[:, :,0] / self.cfg.DATA_STD[0]
        img[:, :,1] = img[:, :,1] / self.cfg.DATA_STD[1]
        img[:, :,2] = img[:, :,2] / self.cfg.DATA_STD[2]

        return img

    def test_time_aug(self,img):
        # rotation three times (90,180,270) + original = 4 times predict
        # compute average as ensemble
        imgs = [img]
        rotations = [2]#[1,2,3]
        for rotation in rotations:
            imgs.append(transform.rotate(img,90 * rotation))

        return imgs

    def __len__(self):
        return len(self.imgs)

if __name__ == '__main__':
    train_cumtom_dataset = DataSet(pharse='train', cfg=cfg)
    train_dataloader = DataLoader(dataset=train_cumtom_dataset,
                                  shuffle=False,
                                  batch_size=1,
                                  num_workers=1)
    train_cumtom_dataset.__getitem__(index=27)