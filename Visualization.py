# coding:utf-8
import numpy as np
from PIL import Image
import os
import tqdm
from scipy import misc

if not os.path.exists(r'./data/train'):
    os.mkdir(r'./data/train')
# if not os.path.exists('./data/test_vis'):
#     os.mkdir('./data/test_vis')

def visualization():
    files = os.listdir(r'F:\AIagriculture\data\train')
    files = [file for file in files if file.endswith('.npy')]
    print(files)
    # files = ['image_2_label.npy']
    if not os.path.exists('./data/train/imgs'):
        os.mkdir(r'./data/train/imgs')
    if not os.path.exists(r'./data/train/masks'):
        os.mkdir(r'./data/train/masks')
    if not os.path.exists(r'./data/train/masks_vis'):
        os.mkdir(r'./data/train/masks_vis')

    for file in files:
        style = False if 'label' in file else True #True for Train
        count = '1' if '1' in file else '2'
        imgs = np.load(os.path.join(r'F:\AIagriculture\data\train',file))
        nums = imgs.shape[0]
        for num in tqdm.tqdm(range(nums)):
            img = Image.fromarray(imgs[num])
            if style:
                img.save('./data/train/imgs/' + count + '_' + str(num) + '.png')
            else:
                img.save('./data/train/masks/' + count + '_' + str(num) + '.png')
                B = imgs[num].copy()  # 蓝色通道 Tobacco
                B[B == 1] = 255
                B[B == 2] = 0
                B[B == 3] = 0
                B[B == 0] = 0

                G = imgs[num].copy()  # 绿色通道 #corn
                G[G == 1] = 0
                G[G == 2] = 255
                G[G == 3] = 0
                G[G == 0] = 0

                R = imgs[num].copy()  # 红色通道 #barley rice
                R[R == 1] = 0
                R[R == 2] = 0
                R[R == 3] = 255
                R[R == 0] = 0

                anno = np.dstack((B, G, R))
                img = Image.fromarray(anno)
                img.save('./data/train/masks_vis/' + count + '_' + str(num) + '.png')
        del imgs

if __name__ == '__main__':
    visualization()






