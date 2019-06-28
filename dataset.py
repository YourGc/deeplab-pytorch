# coding:utf-8
from torch.utils.data import DataLoader,Dataset
import os
from config import cfg
import skimage.io as io
import random
import numpy as np
from imgaug import augmenters as iaa
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
            img,mask = self.aug.augment_images([img,mask])
        # print(mask.shape)
        mask = mask[:,:,np.newaxis]
        img = self.processing(img)
        mask = np.transpose(mask,(2,1,0))
        img = np.transpose(img,(2,1,0))
        return img,self.expand_mask(mask)

    def processing(self,img):
        img = img / 255
        img[:, :0] = img[:,:0] - self.cfg.DATA_MEAN[0]
        img[:, :1] = img[:, :1] - self.cfg.DATA_MEAN[1]
        img[:, :2] = img[:, :2] - self.cfg.DATA_MEAN[2]

        img[:, :0] = img[:, :0] / self.cfg.DATA_STD[0]
        img[:, :1] = img[:, :1] / self.cfg.DATA_STD[1]
        img[:, :2] = img[:, :2] / self.cfg.DATA_STD[2]

        return img

    def expand_mask(self,mask):
        Bsmoke,Corn,Brice = mask.copy(),mask.copy(),mask.copy()
        #print('---')
        #print((mask==0).sum(),(mask==1).sum(),(mask==2).sum(),(mask==3).sum())
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