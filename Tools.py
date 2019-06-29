# coding:utf-8
import os
import numpy as np
import random
import pandas as pd
import skimage.io as io
import tqdm

from PIL import Image
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

def padding_crop():
    #image_3.png, width padding: (32, 33), height padding: (67, 68)
    #Image image_4.png, width padding : (176,176),height padding : (88,88)
    Image.MAX_IMAGE_PIXELS = 100000000000
    SIZE = 512
    STRIDE = 256

    test_path = './data/test'
    img_files = os.listdir(test_path)
    for idx,img_file in enumerate(img_files):
        create_dir(os.path.join(test_path,str(idx+3)))
        img = Image.open(os.path.join(test_path,img_file))
        img = img.convert('RGB')
        img = np.array(img)
        print(img.shape)
        W,H,C = img.shape

        w_count = int(W / SIZE) + 1 if W % SIZE != 0 else int(W / SIZE)
        h_count = int(H / SIZE) + 1 if H % SIZE != 0 else int(W / SIZE)

        w_pad = SIZE * w_count - W
        h_pad = SIZE * h_count - H
        w_pad_left = int(w_pad / 2)
        w_pad_rirght = w_pad - w_pad_left
        h_pad_left = int(h_pad / 2)
        h_pad_right = h_pad - h_pad_left

        print('Image {}, width padding : ({},{}),height padding : ({},{})'.format(img_file,w_pad_left,w_pad_rirght,h_pad_left,h_pad_right))

        img = np.pad(img,(
            (w_pad_left,w_pad_rirght),
            (h_pad_left,h_pad_right),
            (0,0)
        ),mode='constant',constant_values=((0,0),(0,0),(0,0)))
        print(img.shape)

        w_count *= SIZE/STRIDE
        h_count *= SIZE/STRIDE
        print('Croping')
        for i in tqdm.tqdm(range(int(w_count) - 1)):
            for j in range(int(h_count) - 1):
                sub_img = img[i * STRIDE:i * STRIDE + SIZE,j * STRIDE:j * STRIDE + SIZE,:]
                sub_img = Image.fromarray(sub_img)
                sub_img.save(os.path.join(test_path,str(idx+3),str(i) + '_' + str(j) + '.jpg'))

# def IOU(pred,target):
#     #
#     px1,py1,px2,py2 = pred
#     tx1,ty1,tx2,ty2 = target
#
#     left_x = max(tx1,px1)
#     left_y = max(ty1,py1)
#     right_x = min(tx2,px2)
#     right_y = min(ty2,py2)
#
#     if right_y <= left_y or right_x <= left_x : return 0
#     insection = ( right_y - left_y )  * (right_x - left_x)
#     union = (px2 - px1) * (py2-py1) + (tx2-tx1)*(ty2-ty1) - insection
#
#     return union/insection


if __name__ == '__main__':
    a = np.zeros(shape=(4,20000,37000),dtype=np.float32)
    print(a.shape)
    #padding_crop()













