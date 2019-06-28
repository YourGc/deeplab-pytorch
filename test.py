

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import numpy as np
import tqdm

from config import cfg
from lib.datasets.generateData import generate_dataset
from lib.net.generateNet import generate_net
from dataset import Test_DataSet
from lib.net.sync_batchnorm.replicate import patch_replication_callback

from torch.utils.data import DataLoader

def test_net():

	net = generate_net(cfg)
	print('net initialize')
	if cfg.TEST_CKPT is None:
		raise ValueError('test.py: cfg.MODEL_CKPT can not be empty in test period')
	

	print('Use %d GPU'%cfg.TEST_GPUS)
	device = torch.device('cuda')
	if cfg.TEST_GPUS > 1:
		net = nn.DataParallel(net)
		patch_replication_callback(net)
	net.to(device)

	print('start loading model %s'%cfg.TEST_CKPT)
	model_dict = torch.load(cfg.TEST_CKPT,map_location=device)
	net.load_state_dict(model_dict)
	
	net.eval()	
	with torch.no_grad():
		for idx in [1,2]:
			test_dataset = Test_DataSet(idx = idx,cfg =cfg)
			test_dataloader = DataLoader(test_dataset,
									batch_size=1,
									shuffle=False,
									num_workers=2)
			print("-----Test In Image {} -----".format(idx))
			for imgs in tqdm.tqdm(test_dataloader):
				results = []
				for img in imgs:
					output = net(img)
					results.append(output)

				final_result = AverageResult(results).detach().cpu().numpy()

				concate(final_result)

def concate(patch):
	pass

def AverageResult(results):

	C,H,W = results[0].shape
	final_result = torch.FloatTensor(C,H,W).zero_().cuda()
	for result in results:
		final_result += result

	return final_result/4

if __name__ == '__main__':
	test_net()


