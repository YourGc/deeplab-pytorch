

import torch
import torch.nn as nn
import argparse
import tqdm
import numpy as np
import os
from torch.autograd import Variable
from config import cfg
from lib.datasets.generateData import generate_dataset
from lib.net.generateNet import generate_net
from dataset import Test_DataSet
from lib.net.sync_batchnorm.replicate import patch_replication_callback
from PIL import Image
from torch.utils.data import DataLoader
from Tools import create_dir

Image.MAX_IMAGE_PIXELS = 100000000000
def test_net(args):

	net = generate_net(cfg)
	print('net initialize')

	print('Use %d GPU'%cfg.TEST_GPUS)
	device = torch.device('cuda')
	if cfg.TEST_GPUS > 1:
		net = nn.DataParallel(net)
		patch_replication_callback(net)
	net.to(device)


	print('start loading model %s'%args.model_path)
	model_dict = torch.load(args.model_path,map_location=device)
	net.load_state_dict(model_dict)
	
	net.eval()
	STRIDE = 256
	SIZE = 512

	create_dir(args.save_dir)
	with torch.no_grad():
		for idx in [3,4]:#img idx
			max_w = max_h = None
			w_pad = h_pad = None

			if idx == 3:
				max_w,max_h = 78,146 #STRIDE = 256,SIEZ = 512
				w_pad = (32, 33)
				h_pad = (67, 68)
			elif idx == 4:
				max_w,max_h = 114,102 #STRIDE = 256,SIEZ = 512
				w_pad = (176, 176)
				h_pad = (88, 88)
			test_dataset = Test_DataSet(idx = idx,cfg =cfg)
			test_dataloader = DataLoader(test_dataset,
									batch_size=1,
									shuffle=False,
									num_workers=2)

			mask = np.zeros(shape=(4,max_w * STRIDE,max_h * STRIDE),dtype=np.float32)
			#print(mask.shape)
			create_dir(os.path.join(args.save_dir,str(idx) + '_tmp'))
			print("-----Test In Image {} -----".format(idx))

			for imgs,name in tqdm.tqdm(test_dataloader):
				results = []
				# for img in imgs:
				# 	img = Variable(img).float().cuda()
				# 	output = net(img)
				# 	results.append(output.squeeze())
				# result = AverageResult(results)#.detach().cpu().numpy()
				img = Variable(imgs).float().cuda()
				output = net(img)
				result = output.squeeze().detach().cpu().numpy()
				#result.tofile(os.path.join(args.save_dir,str(idx) + '_tmp',name[0].strip('jpg') + 'npy'))
				# print(result.shape,mask.shape)
				w_idx,h_idx = str(name[0]).strip('.jpg').split('_')
				w_idx,h_idx = int(w_idx), int(h_idx)
				mask[:,w_idx * STRIDE:w_idx * STRIDE + SIZE ,h_idx * STRIDE:h_idx * STRIDE+SIZE] = \
					mask[:, w_idx * STRIDE:w_idx * STRIDE + SIZE, h_idx * STRIDE:h_idx * STRIDE + SIZE] + \
					result
			del test_dataloader
			del test_dataset
			#fix mask 取均值防止重复计算
			mask[:,STRIDE:-STRIDE,:] /=2
			mask[:,:,STRIDE:-STRIDE] /= 2
			mask = mask[:,w_pad[0] : -w_pad[1],h_pad[0]:-h_pad[1]]
			#通道整合
			# mask.tofile('mask.npy')# 以防万一

			# for w in tqdm.tqdm(range(max_w)):
			# 	for h in range(max_h):
			# 		mask[0,w:(w+1) * SIZE,h:(h+1)*SIZE] = np.argmax(mask[:,w:(w+1) * SIZE,h:(h+1)*SIZE],axis=0)
			#
			# mask = mask[0,:,:]
			mask = np.argmax(mask,axis=0) #内存溢出
			mask = Image.fromarray(np.uint8(mask))
			create_dir(args.save_dir)
			mask.save(os.path.join(args.save_dir,str(idx) + '.png'))
# def concate(args):
# 	#3_tmp,4_tmp
#
# 	for dir in ['3_tmp','4_tmp']:
# 		files = os.listdir(os.path.join(args.save_dir,dir))
# 		files = sorted(files,key=lambda x:int(str(x).strip('.npy').split('_')[0]) * 10 \
# 								  + int(str(x).strip('.npy').split('_')[1]))
# 		max_w = max_h = None
# 		w_pad = h_pad = None
#
# 		if idx == 3:
# 			max_w, max_h = 78, 146  # STRIDE = 256,SIEZ = 512
# 			w_pad = (32, 33)
# 			h_pad = (67, 68)
# 		elif idx == 4:
# 			max_w, max_h = 114, 102  # STRIDE = 256,SIEZ = 512
# 			w_pad = (176, 176)
# 			h_pad = (88, 88)

def AverageResult(results):
	#旋转角度逆时针[0，90，180，270]
	C,H,W = results[0].shape
	final_result = np.zeros(shape=(C,H,W)) #torch.FloatTensor(C,H,W).zero_().cuda()
	for i,result in enumerate(results):
		#反转回来
		result = result.detach().cpu().numpy()
		final_result += np.rot90(result,i * -1,(1,2))

	return final_result/4

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description="Test")
	parser.add_argument('--model_path', type=str, required=True, help='model_path')
	parser.add_argument('--save_dir',type=str,required=True,help='save_dir')
	args = parser.parse_args()
	test_net(args)


