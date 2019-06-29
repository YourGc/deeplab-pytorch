# coding:utf-8
'''
    TO split images into small

    orignal data tree
    data :
        train_ori:
            image_1.png
            image_2.png
            image_1_label.png
            image_2_label.png
        test:
            image_3.png
            iamge_4.png
'''

import os
import cv2
from PIL import Image
import numpy as np



#read images
Image.MAX_IMAGE_PIXELS = 100000000000
img = Image.open(r'./data/train_ori/image_1.png')   # 注意修改img路径
img = np.asarray(img)
print(img.shape)

anno_map = Image.open('./data/train_ori/image_1_label.png')   # 注意修改label路径
anno_map = np.asarray(anno_map)
print(anno_map.shape)

cimg = cv2.resize(img, None, fx= 0.1, fy=0.1)
cimg = cv2.cvtColor(cimg, cv2.COLOR_RGB2BGR)
cv2.imwrite('./data/train/train_image_1.png', cimg, [int(cv2.IMWRITE_JPEG_QUALITY),100])   # 注意修改 可视化img 的路径

# cimg = cv2.resize(img, None, fx= 0.1, fy=0.1)
# cimg = cv2.cvtColor(cimg, cv2.COLOR_RGB2BGR)
# cv2.imwrite('./data/train/train_image_1.png', cimg, [int(cv2.IMWRITE_JPEG_QUALITY),100])   # 注意修改 可视化img 的路径


#visualization
B = anno_map.copy()   # 蓝色通道
B[B == 1] = 255
B[B == 2] = 0
B[B == 3] = 0
B[B == 0] = 0

G = anno_map.copy()   # 绿色通道
G[G == 1] = 0
G[G == 2] = 255
G[G == 3] = 0
G[G == 0] = 0

R = anno_map.copy()   # 红色通道
R[R == 1] = 0
R[R == 2] = 0
R[R == 3] = 255
R[R == 0] = 0

anno_vis = np.dstack((B,G,R))
anno_vis = cv2.resize(anno_vis, None, fx= 0.1, fy=0.1)
cv2.imwrite('./data/train/train_image_1_label.png', anno_vis)  # 注意修改 可视化label 的路径



#crop
unit_size= 512   # 窗口大小

length, width = img.shape[0], img.shape[1]
x1, x2, y1, y2 = 0, unit_size ,0 ,unit_size
Img = [] # 保存小图的数组
Label = []  # 保存label的数组
while(x1 < length):
    #判断横向是否越界
    if x2 > length:
        x2 , x1 = length , length - unit_size

    while(y1 < width):
        if y2 > width:
            y2 , y1  = width , width - unit_size
        im = img[x1:x2, y1:y2, :]
        if 255 in im[:,:,-1]:    # 判断裁剪出来的小图中是否存在有像素点
            Img.append(im[:,:,0:3])   # 添加小图
            Label.append(anno_map[x1:x2, y1:y2])   # 添加label

        if y2 == width: break

        y1 += unit_size
        y2 += unit_size

    if x2 == length: break

    y1,y2 = 0 , unit_size
    x1 += unit_size
    x2 += unit_size

Img = np.array(Img)
Label = np.array(Label)

print(Img.shape)
print(Label.shape)



#save
np.save('./data/train/image_1.npy', Img)    # 注意修改 npy-Img 的路径
np.save('./data/train/image_1_label.npy', Label)   # 注意修改 npy-label 的路径

#
# # !/usr/bin/env python
# # coding:utf-8
# """
# Name    : png_process.py
# Author  : .mat
# Github  : www.github.com/kiclent/tianchi_jinwei_AI2019.git
# Contect : kiclent@yahoo.com
# Time    : 2019-06-21 10:14
# Desc    : PNG 大图切割程序。临时恶补的PNG解码知识，难免会有错误，希望大家指出。
#           后期如果有时间的话，可能会不定期的更新一些本次比赛的代码。
#
#           参考资料 :
#           https://www.cnblogs.com/wzjhoutai/p/7146232.html
#           https://www.w3.org/TR/PNG/#5DataRep
#           https://docs.python.org/3/library/zlib.html
#           https://github.com/jarvisteach/tkinter-png
#           https://blog.csdn.net/chijingjing/article/details/80186018
# """
#
# import os
# import struct
# import zlib
# import numpy as np
# from PIL import Image
# from time import time as tc
#
# _BYTE_HEAD = 8  # PNG 文件头占用空间
# _BYTE_CHUNK_LENGTH = 4  # Chunk Length 占用空间
# _BYTE_CHUNK_TYPE_CODE = 4  # Chunk Type Code 占用空间
# _BYTE_CHUNK_CRC32 = 4  # CRC 校验码占用空间
#
#
# def png_read_head(fp):
#     """
#     :param fp:
#     :return:
#     """
#     png_head = fp.read(_BYTE_HEAD)
#     return png_head
#
#
# def png_read_chunk(fp):
#     """
#     根据数据块定义的格式读取并解析数据
#     :param
#         fp:
#     :return:
#         chunk_data: chunk 数据
#         chunk_type_code : chunk 的标识符
#     """
#
#     # Chunk 长度读取
#     chunk_length = fp.read(_BYTE_CHUNK_LENGTH)
#
#     # 转换为 32 位无符号整型
#     chunk_length = struct.unpack('!I', chunk_length)[0]
#
#     # Chunk Type Code
#     chunk_type_code = fp.read(_BYTE_CHUNK_TYPE_CODE)
#
#     # Chunk 的数据读取
#     chunk_data = fp.read(chunk_length)
#
#     # CRC 校验
#     chunk_crc32 = fp.read(_BYTE_CHUNK_CRC32)
#     chunk_crc32 = struct.unpack('!I', chunk_crc32)[0]
#     check_crc32 = zlib.crc32(chunk_type_code + chunk_data)
#
#     # print(check_crc32, chunk_crc32)
#     assert check_crc32 == chunk_crc32, print('CRC校验出错')
#
#     # byte --> str
#     chunk_type_code = str(chunk_type_code, encoding='utf-8')
#
#     return chunk_data, chunk_type_code
#
#
# def png_ihdr_decode(chunk_data):
#     """
#     根据定义解析IHDR
#     :param chunk_data:
#     :return:
#     """
#
#     image_info = {}
#     image_info['width'] = struct.unpack('!I', chunk_data[:4])[0]
#     image_info['height'] = struct.unpack('!I', chunk_data[4:8])[0]
#     image_info['bitDepth'] = int(chunk_data[8])
#     image_info['colorType'] = int(chunk_data[9])
#     image_info['compressionMethod'] = int(chunk_data[10])
#     image_info['filterMethod'] = int(chunk_data[11])
#     image_info['interlaceMethod'] = int(chunk_data[12])
#
#     return image_info
#
#
# def png_sub(png_file, save_dir, block_size=(512, 512)):
#     """
#     解压 idat 数据
#     :param png_file:
#     :param save_dir:
#     :param block_size: (height, width)
#     :return:
#     """
#
#     # 创建保存目录
#     if not os.path.exists(save_dir): os.makedirs(save_dir)
#
#     # 解压工具初始化
#     decoder = zlib.decompressobj(zlib.MAX_WBITS | 32)
#
#     # 读取文件
#     with open(png_file, 'rb') as fp:
#
#         # 读取PNG文件签名信息
#         png_head = png_read_head(fp)
#         print(png_head.hex().upper())
#
#         # 读取IHDR数据块
#         chunk_data, chunk_type_code = png_read_chunk(fp)
#
#         # 解析IHDR数据块获得图片信息
#         image_info = png_ihdr_decode(chunk_data)
#
#         # 只针对本题的PNG图片格式设置
#         if image_info['colorType'] == 4 and image_info['bitDepth'] == 8:
#             _BYTE_COLOR_TYPE = 2
#             mode = 'L'
#         elif image_info['colorType'] == 6 and image_info['bitDepth'] == 8:
#             _BYTE_COLOR_TYPE = 4
#             mode = 'RGBA'
#         else:
#             _BYTE_COLOR_TYPE = 1
#             mode = 'L'
#
#         bytes_stream = b''
#         row_cnt = 0
#         idat_id = 0
#         while chunk_type_code != 'IEND':
#
#             # 读取数据块
#             chunk_data, chunk_type_code = png_read_chunk(fp)
#
#             if chunk_type_code == 'IDAT':
#                 # 解压
#                 chunk_data = decoder.decompress(chunk_data)
#                 bytes_stream += chunk_data
#                 idat_id += 1
#
#             if chunk_type_code == 'IEND' and len(bytes_stream) > 0:
#                 block_height = len(bytes_stream) // (image_info['width'] * _BYTE_COLOR_TYPE + 1)
#             else:
#                 block_height = block_size[0]
#
#             if len(bytes_stream) >= block_height * (image_info['width'] * _BYTE_COLOR_TYPE + 1):
#                 #  get filter type
#                 filter_type = int(bytes_stream[0])
#                 # print(filter_type)
#
#                 row_block = bytes_stream[:block_height * (image_info['width'] * _BYTE_COLOR_TYPE + 1)]
#                 row_block = np.array(list(row_block), np.uint8)
#                 row_block = row_block.reshape((block_height, image_info['width'] * _BYTE_COLOR_TYPE + 1))
#                 if _BYTE_COLOR_TYPE == 1:
#                     row_block = row_block[:, 1:].reshape((block_height, image_info['width']))
#                 else:
#                     row_block = row_block[:, 1:].reshape((block_height, image_info['width'], _BYTE_COLOR_TYPE))
#
#                 # 只针对 sub 进行解码
#                 if filter_type == 1:
#                     for col in range(image_info['width'] - 1):
#                         row_block[:, col + 1] += row_block[:, col]
#
#                 bytes_stream = bytes_stream[block_height * (image_info['width'] * _BYTE_COLOR_TYPE + 1):]
#
#                 for i in range(0, image_info['width'] - block_size[1], block_size[1]):
#                     sub_image = row_block[:, i:i + block_size[1]]
#                     sub_image = Image.fromarray(sub_image, mode=mode)
#                     sub_image.save(
#                         os.path.join(save_dir, 'img_{}_{}_.png'.format(row_cnt, int(np.ceil(i / block_size[1])))))
#                     print('block ({}, {})'.format(row_cnt, int(np.ceil(i / block_size[1]))))
#
#                 row_cnt += 1
#
#         print('{} 切割完成！'.format(png_file))
#
#
# def check_png(png_file):
#     """
#     查看PNG图像基本信息
#     :param png_file:
#     :return:
#     """
#
#     chunk_type = {}
#     with open(png_file, 'rb') as fp:
#
#         # 读取PNG文件签名信息
#         png_head = png_read_head(fp)
#         print(png_head.hex().upper())
#
#         # 读取IHDR数据块
#         chunk_data, chunk_type_code = png_read_chunk(fp)
#         chunk_type[chunk_type_code] = 1
#
#         # 解析IHDR数据块获得图片信息
#         image_info = png_ihdr_decode(chunk_data)
#
#         while chunk_type_code != 'IEND':
#
#             # 读取数据块
#             chunk_data, chunk_type_code = png_read_chunk(fp)
#
#             if chunk_type_code not in chunk_type.keys():
#                 chunk_type[chunk_type_code] = 1
#             else:
#                 chunk_type[chunk_type_code] += 1
#
#         print(image_info)
#         print(chunk_type)
#
#
# if __name__ == "__main__":
#     tic = tc()
#
#     block_size = (512, 512)
#
#     # ------------- train data ----------------
#     # train image_1.png
#     png_file = './data/train_ori/image_1.png'
#     # check_png(png_file)
#     os.makedirs('./data/train/train_1')
#     png_sub(png_file, './data/train/train_1', block_size)
#     print('train image_1.png', tc() - tic)
#
#     # train image_2.png
#     png_file = './data/train_ori/image_2.png'
#     # check_png(png_file)
#     os.makedirs('./data/train/train_2')
#     png_sub(png_file, './data/sub/train_2', block_size)
#     print('train image_2.png', tc() - tic)
#
#     # train image_1_label.png
#     png_file = './data/train_ori/image_1_label.png'
#     # check_png(png_file)
#     os.makedirs('./data/sub/train_1_label')
#     png_sub(png_file, './data/sub/train_1_label', block_size)
#     print('train image_1_label.png', tc() - tic)
#
#     # train image_2_label.png
#     png_file = './data/train_ori/image_2_label.png'
#     # check_png(png_file)
#     os.makedirs('./data/sub/train_2_label')
#     png_sub(png_file, './data/sub/train_2_label', block_size)
#     print('train image_2_label.png', tc() - tic)
#
#     # ------------- test data ----------------
#     # test image_3.png
#     png_file = './data/test_ori/image_3.png'
#     # check_png(png_file)
#     os.makedirs('./data/sub/test_a_3')
#     png_sub(png_file, './data/sub/test_a_3', block_size)
#     print('test image_3.png', tc() - tic)
#
#     # test image_4.png
#     png_file = './data/test_ori/image_4.png'
#     # check_png(png_file)
#     os.makedirs('./data/sub/test_a_4')
#     png_sub(png_file, './data/sub/test_a_4', block_size)
#     print('test image_4.png', tc() - tic)
