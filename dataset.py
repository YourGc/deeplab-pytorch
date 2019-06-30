# coding:utf-8
from torch.utils.data import DataLoader,Dataset
import os
from config import cfg
from skimage import io, transform
import random
import numpy as np
from imgaug import augmenters as iaa
import imgaug as ia
import torch
random.seed(666)


class DataSet(Dataset):
    def __init__(self,pharse,cfg,root = './data'):
        super(Dataset,self).__init__()
        self.pharse = pharse
        self.img_path = root + r'/imgs'
        self.lable_path = root + r'/masks'
        self.files = self.load_file()
        # self.files = [self.files[i] for i in range(50)]
        self.cfg = cfg
        self.aug = iaa.Sequential([
            iaa.Fliplr(0.5),
            iaa.Flipud(0.5)],random_state=666
        )
        random.shuffle(self.files)

    def __getitem__(self, index):
        img = io.imread(os.path.join(self.img_path,self.files[index]))
        mask = io.imread(os.path.join(self.lable_path,self.files[index]))
        img = np.array(img)
        mask = np.array(mask)
        if self.cfg.DATA_AUG:
            segmap = ia.SegmentationMapOnImage(mask, shape=mask.shape, nb_classes=cfg.MODEL_NUM_CLASSES)
            self.aug.to_deterministic()
            img = self.aug.augment_image(img)
            mask = self.aug.augment_segmentation_maps([segmap])[0].get_arr_int().astype(np.uint8)
        # print(mask.shape)
        mask = mask[:,:,np.newaxis]
        img = self.processing(img)
        mask = np.transpose(mask,(2,1,0))
        img = np.transpose(img,(2,1,0))
        # img,mask = torch.Tensor(img) ,torch.Tensor(mask)
        mask = self.expand_mask(mask)
        # img = torch.Tensor(img).float()
        return img,mask

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

    def expand_mask(self,mask):
        # H,W = mask.shape
        # mask = torch.Tensor(mask).long()
        # masks = torch.Tensor(self.cfg.MODEL_NUM_CLASSES,H,W).zero_().float()
        # masks = masks.scatter_(1, mask, 1.)
        #print('---')
        Bsmoke,Corn,Brice = mask.copy(),mask.copy(),mask.copy()
        # print((mask==0).sum(),(mask==1).sum(),(mask==2).sum(),(mask==3).sum())
        Bsmoke[Bsmoke!=1] = 0
        Bsmoke[Bsmoke==1] = 1

        Corn[Corn!=2] = 0
        Corn[Corn==2] = 1

        Brice[Brice  != 3 ] = 0
        Brice[Brice  == 3] = 1

        mask[mask != 0] = 2
        mask[mask == 0] =1
        mask[mask == 2] = 0
        #print(mask.sum(),Bsmoke.sum(),Corn.sum(),Brice.sum())
        masks = np.vstack((mask,Bsmoke,Corn,Brice))
        # print(masks.sum())
        return masks

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
        imgs = self.test_time_aug(img)
        imgs = [np.array(img)for img in imgs]
        imgs = [self.processing(img) for img in imgs]
        imgs = [np.transpose(img,(2,1,0)) for img in imgs]
        return imgs,img_name

    def processing(self,img):
        img = img / 255
        img[:, :0] = img[:,:0] - self.cfg.DATA_MEAN[0]
        img[:, :1] = img[:, :1] - self.cfg.DATA_MEAN[1]
        img[:, :2] = img[:, :2] - self.cfg.DATA_MEAN[2]

        img[:, :0] = img[:, :0] / self.cfg.DATA_STD[0]
        img[:, :1] = img[:, :1] / self.cfg.DATA_STD[1]
        img[:, :2] = img[:, :2] / self.cfg.DATA_STD[2]

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
                                  shuffle=True,
                                  batch_size=cfg.TRAIN_BATCHES,
                                  num_workers=cfg.DATA_WORKERS)
    print(len(train_cumtom_dataset))
    for i,data in enumerate(train_dataloader):
        print(i)
        pass