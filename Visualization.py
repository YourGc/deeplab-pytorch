# coding:utf-8
import numpy as np
from PIL import Image
import os
import tqdm
from scipy import misc

if not os.path.exists('./data/train_vis'):
    os.mkdir('./data/train_vis')
if not os.path.exists('./data/test_vis'):
    os.mkdir('./data/test_vis')

def visualization():
    files = os.listdir(os.path.join(r'./data','train'))
    files = [file for file in files if file.endswith('.npy')]
    if not os.path.exists('./data/train_vis/imgs'):
        os.mkdir('./data/train_vis/imgs')
    if not os.path.exists('./data/train_vis/masks'):
        os.mkdir('./data/train_vis/masks')

    for file in files:
        style = False if 'label' in file else True #True for Train
        count = '1' if '1' in file else '2'
        imgs = np.load(os.path.join('./data','train',file))
        nums = imgs.shape[0]
        for num in tqdm.tqdm(range(nums)):
            img = Image.fromarray(imgs[num])
            if style:
                img.save('./data/train_vis/imgs/' + count + '_' + str(num) + '.jpg')
            else:
                # B = imgs[num].copy()  # 蓝色通道
                # B[B == 1] = 255
                # B[B == 2] = 0
                # B[B == 3] = 0
                # B[B == 0] = 0
                #
                # G = imgs[num].copy()  # 绿色通道
                # G[G == 1] = 0
                # G[G == 2] = 255
                # G[G == 3] = 0
                # G[G == 0] = 0
                #
                # R = imgs[num].copy()  # 红色通道
                # R[R == 1] = 0
                # R[R == 2] = 0
                # R[R == 3] = 255
                # R[R == 0] = 0

                # anno = np.dstack((R, G, B))
                # img = Image.fromarray(anno)
                img.save('./data/train_vis/masks/' + count + '_' + str(num) + '.png')

if __name__ == '__main__':
    visualization()






