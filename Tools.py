# coding:utf-8
import os
import numpy as np
import random
import pandas as pd
import skimage.io as io
import tqdm
random.seed(666)

def read_train_txt():
    path = './data/train.txt'
    with open(path, 'r') as f:
        names = f.readlines()
    names = [name.strip() for name in names]
    return names

def create_dir(path):
    if not os.path.exists(path):
        os.mkdir(path)

def train_val_split(split = 0.2):
    if os.path.exists('./data/train.txt') and os.path.exists('./data/val.txt'):
        return

    files = os.listdir('./data/imgs')
    val = random.Random().sample(files,int(split * len(files)))
    train = [file for file in files if file not in val]

    val = pd.DataFrame(val,columns = ['name'])
    train = pd.DataFrame(train,columns = ['name'])

    val.to_csv('./data/val.txt',header=False,index=False)
    train.to_csv('./data/train.txt',header=False,index=False)

def compute_mean():
    # R: 0.51135, G: 0.50602, B: 0.44164
    names = read_train_txt()

    R,G,B = 0.0,0.0,0.0
    n = len(names)
    count = 512 * 512
    for name in tqdm.tqdm(names):
        img = io.imread(os.path.join('./data/imgs',name))
        img = np.array(img)
        R += img[:,:,0].sum() / count
        G += img[:,:,1].sum() / count
        B += img[:,:,2].sum() / count
    R/=n
    G/=n
    B/=n

    normed_R = R / 255
    normed_G = G / 255
    normed_B = B / 255
    print("mean_R:{:.5},mean_G:{:.5},mean_B:{:.5}".format(normed_R,normed_G,normed_B))

def compute_std():
    #std_R: 0.24587, std_G: 0.21886, std_B: 0.21088
    names = read_train_txt()

    R, G, B = 0.0, 0.0, 0.0
    mean = [0.51135* 255,0.50602* 255,0.44164* 255]
    n = len(names)
    count = 512 * 512
    for name in tqdm.tqdm(names):
        img = io.imread(os.path.join('./data/imgs', name))
        img = np.array(img)
        R += ((img[:, :, 0] - mean[0]) ** 2 ).sum() / count
        G += ((img[:, :, 1] - mean[1]) ** 2 ).sum() / count
        B += ((img[:, :, 2] - mean[2]) ** 2 ).sum() / count
    R /= n
    G /= n
    B /= n

    normed_R = R**0.5 / 255
    normed_G = G**0.5 / 255
    normed_B = B**0.5 / 255

    print("std_R:{:.5},std_G:{:.5},std_B:{:.5}".format(normed_R, normed_G, normed_B))

def compute_class_weight():
    #bg:0.694139,Bsmoke:0.088105,Corn:0.072427,Brice:0.145329
    names = read_train_txt()
    bg,Bsmoke,Corn,Brice = 0,0,0,0
    total = len(names) * 512 * 512
    for name in tqdm.tqdm(names):
        img = io.imread(os.path.join('./data/masks', name))
        img = np.array(img)
        bg += (img == 0).sum()
        Bsmoke += (img == 1).sum()
        Corn += (img == 2).sum()
        Brice += (img == 3).sum()
    assert bg + Bsmoke + Corn + Brice == total , print('ERROR')

    print('bg:{:.6f},Bsmoke:{:.6f},Corn:{:.6f},Brice:{:.6f}'.format(bg/total,Bsmoke/total,Corn/total,Brice/total))

if __name__ == '__main__':
    train_val_split()













