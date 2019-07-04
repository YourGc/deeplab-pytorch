
import torch
import argparse
import os
import sys
import cv2
import time

class Configuration():
	def __init__(self):
		self.ROOT_DIR = os.path.abspath(os.path.dirname("__file__"))
		self.EXP_NAME = 'deeplabv3'

		self.DATA_NAME = 'VOC2012'
		self.DATA_AUG = False
		self.DATA_WORKERS = 4
		self.DATA_SIZE = 512
		self.DATA_RESCALE = 512
		self.DATA_RANDOMCROP = 512
		self.DATA_RANDOMROTATION = 0
		self.DATA_RANDOMSCALE = 2
		self.DATA_RANDOM_H = 10
		self.DATA_RANDOM_S = 10
		self.DATA_RANDOM_V = 10
		self.DATA_RANDOMFLIP = 0.5
		self.DATA_MEAN = [0.51135,0.50602,0.44164]
		self.DATA_STD = [0.24587,0.21886,0.21088]

		self.MODEL_NAME = 'deeplabv3plus'
		self.MODEL_BACKBONE = 'res101_atrous'
		self.MODEL_OUTPUT_STRIDE = 16
		self.MODEL_ASPP_OUTDIM = 256
		self.MODEL_SHORTCUT_DIM = 48
		self.MODEL_SHORTCUT_KERNEL = 1
		self.MODEL_NUM_CLASSES = 4
		self.MODEL_SAVE_DIR = os.path.join(self.ROOT_DIR,'model',self.EXP_NAME)

		self.TRAIN_LR = 0.001
		self.TRAIN_LR_GAMMA = 0.1
		self.TRAIN_MOMENTUM = 0.9
		self.TRAIN_WEIGHT_DECAY = 0.0001
		self.TRAIN_BN_MOM = 0.0003
		self.TRAIN_POWER = 0.9
		self.TRAIN_GPUS = 4
		self.TRAIN_BATCHES = 16
		self.TRAIN_SHUFFLE = True
		self.TRAIN_MINEPOCH = 20
		self.TRAIN_EPOCHS = 50
		self.TRAIN_LOSS_LAMBDA = 0
		self.TRAIN_TBLOG = True
		self.TRAIN_CKPT = os.path.join(self.ROOT_DIR,'model/deeplabv3/deeplabv3plus_res101_atrous_VOC2012_epoch20.pth')

		self.LOG_DIR = os.path.join(self.ROOT_DIR,'log',self.EXP_NAME)

		self.PRINT_FRE = 1
		self.SAVE_FRE = 5

		self.TEST_MULTISCALE = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75]
		self.TEST_FLIP = True
		self.TEST_CKPT = os.path.join(self.ROOT_DIR,'model/deeplabv3/deeplabv3plus_res101_atrous_VOC2012_epoch20.pth')
		self.TEST_GPUS = 4
		self.TEST_BATCHES = 16

		self.__check()
		self.__add_path(os.path.join(self.ROOT_DIR, 'lib'))

	def __check(self):
		if not torch.cuda.is_available():
			raise ValueError('config.py: cuda is not avalable')
		if self.TRAIN_GPUS == 0:
			raise ValueError('config.py: the number of GPU is 0')
		#if self.TRAIN_GPUS != torch.cuda.device_count():
		#	raise ValueError('config.py: GPU number is not matched')
		if not os.path.isdir(self.LOG_DIR):
			os.makedirs(self.LOG_DIR)
		if not os.path.isdir(self.MODEL_SAVE_DIR):
			os.makedirs(self.MODEL_SAVE_DIR)

	def __add_path(self, path):
		if path not in sys.path:
			sys.path.insert(0, path)



cfg = Configuration() 	
