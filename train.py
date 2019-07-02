
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import os
import sys
import numpy as np
import tqdm
import torch.nn.functional as F
import matplotlib.pyplot as plt

from torch.autograd import Variable
from config import cfg
from lib.net.generateNet import generate_net
import torch.optim as optim
from PIL import Image
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from lib.net.loss import MaskLoss
from lib.net.sync_batchnorm.replicate import patch_replication_callback
from dataset import DataSet
from tensorboardX import SummaryWriter
from lib.net.refinenet import refinenet

esp = 1e-8
torch.backends.cudnn.benchmark = True

def train_net():
	train_cumtom_dataset = DataSet(pharse='train',cfg=cfg)
	train_dataloader = DataLoader(dataset=train_cumtom_dataset,
                            shuffle=False,
                            batch_size=cfg.TRAIN_BATCHES,
                            num_workers=cfg.DATA_WORKERS)

	val_cumtom_dataset = DataSet(pharse='val',cfg=cfg)
	val_dataloader = DataLoader(dataset=val_cumtom_dataset,
							shuffle=True,
							batch_size=cfg.TEST_BATCHES,
							num_workers=cfg.DATA_WORKERS)
	print('train dataset : {} ,with batch size :{}'.format(len(train_cumtom_dataset),cfg.TRAIN_BATCHES))
	print('val dataset : {} ,with batch size :{}'.format(len(val_cumtom_dataset), cfg.TEST_BATCHES))
	# dataset = generate_dataset(cfg.DATA_NAME, cfg, 'train', cfg.DATA_AUG)
	# dataloader = DataLoader(dataset,
	# 			batch_size=cfg.TRAIN_BATCHES,
	# 			shuffle=cfg.TRAIN_SHUFFLE,
	# 			num_workers=cfg.DATA_WORKERS,
	# 			drop_last=True)
	
	net = generate_net(cfg)

	#net =refinenet(cfg,False)
	#net.apply(weights_init)


	print('Use %d GPU'%cfg.TRAIN_GPUS)
	device = torch.device(0)
	if cfg.TRAIN_GPUS > 1:
		net = nn.DataParallel(net)
		patch_replication_callback(net)
	net.to(device)		

	if cfg.TRAIN_CKPT:
		pretrained_dict = torch.load(cfg.TRAIN_CKPT)
		net_dict = net.state_dict()
		pretrained_dict = {k: v for k, v in pretrained_dict.items() if (k in net_dict) and (v.shape==net_dict[k].shape)}
		net_dict.update(pretrained_dict)
		net.load_state_dict(net_dict)
		# net.load_state_dict(torch.load(cfg.TRAIN_CKPT),False)
	
	criterion = MaskLoss()
	# optimizer = optim.SGD(
	# 	params = [
	# 		{'params': get_params(net.module,key='1x'), 'lr': cfg.TRAIN_LR},
	# 		{'params': get_params(net.module,key='10x'), 'lr': 10*cfg.TRAIN_LR}
	# 	],
	# 	momentum=cfg.TRAIN_MOMENTUM
	# )
	optimizer = optim.SGD(
		lr=cfg.TRAIN_LR,
		params = net.parameters(),
		momentum=cfg.TRAIN_MOMENTUM,
		weight_decay=cfg.TRAIN_WEIGHT_DECAY
	)
	# scheduler = optim.lr_scheduler.(optimizer,gamma=cfg.TRAIN_LR_GAMMA)
	#scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=cfg.TRAIN_LR_MST, gamma=cfg.TRAIN_LR_GAMMA, last_epoch=-1)
	itr = cfg.TRAIN_MINEPOCH * len(train_dataloader)
	max_itr = cfg.TRAIN_EPOCHS * len(train_dataloader)
	running_loss = 0.0

	tblogger = SummaryWriter()
	# print('asdas')
	for epoch in range(cfg.TRAIN_MINEPOCH, cfg.TRAIN_EPOCHS):
		# scheduler.step()
		net.train()
		now_lr = adjust_lr(optimizer, epoch)
		avaliable_unions = [esp for _ in range(cfg.MODEL_NUM_CLASSES)]
		avaliable_insections = [0.0 for _ in range(cfg.MODEL_NUM_CLASSES)]
		for i, data in enumerate(train_dataloader):
			img, mask = data
			img = Variable(img).float().cuda()
			mask = Variable(mask).long().cuda()
			#print(mask.shape, type(mask))
			optimizer.zero_grad()
			output = net(img)
			loss = criterion(output, mask)
			compute_iou(output,mask,avaliable_unions,avaliable_insections)

			loss.backward()
			optimizer.step()

			cur_loss = loss.item()
			# print(cur_loss)

			# net.eval()
			# output = net(img)
			# eval_loss = criterion(output,mask).item()
			# print(eval_loss)
			# net.train()

			running_loss += cur_loss
			if i!=0 and i % cfg.PRINT_FRE == 0:
				print('epoch:{}/{}\tbatch:{}/{}\tlr:{:.6f}\tloss:{:.6f}\tBsmoke:{:.6f}\tCorn:{:.6f}\tBrice:{:.6f}\tBG:{:.6f}'.format(
					epoch, cfg.TRAIN_EPOCHS, i, len(train_dataloader),
					now_lr, running_loss/(i+1) , avaliable_insections[1]/avaliable_unions[1], avaliable_insections[2]/avaliable_unions[2]
					 , avaliable_insections[3] / avaliable_unions[3],avaliable_insections[0]/avaliable_unions[0]))

		tblogger.add_scalars('loss', {'train':running_loss / len(train_dataloader)}, epoch )
		tblogger.add_scalars('bg', {'train':avaliable_insections[0]/avaliable_unions[0]}, epoch )
		tblogger.add_scalars('Bsmoke', {'train':avaliable_insections[1]/avaliable_unions[1]}, epoch )
		tblogger.add_scalars('Corn', {'train':avaliable_insections[2]/avaliable_unions[2]}, epoch )
		tblogger.add_scalars('Brice', {'train':avaliable_insections[3]/avaliable_unions[3]}, epoch )

		running_loss = 0.0
			
		if epoch != 0 and epoch % cfg.SAVE_FRE == 0:
			save_path = os.path.join(cfg.MODEL_SAVE_DIR,'%s_%s_%s_epoch%d.pth'%(cfg.MODEL_NAME,cfg.MODEL_BACKBONE,cfg.DATA_NAME,epoch))
			torch.save(net.state_dict(), save_path)
			print('%s has been saved'%save_path)

		print('evalution at epoch {}'.format(epoch))
		eval(net,val_dataloader,criterion,tblogger,epoch)
		
	save_path = os.path.join(cfg.MODEL_SAVE_DIR,'%s_%s_%s_epoch%d_all.pth'%(cfg.MODEL_NAME,cfg.MODEL_BACKBONE,cfg.DATA_NAME,cfg.TRAIN_EPOCHS))		
	torch.save(net.state_dict(),save_path)
	if cfg.TRAIN_TBLOG:
		tblogger.close()
	print('%s has been saved'%save_path)
	print('train finished!')

def eval(net,dataloader,criterion,logger,epoch):
	# net.eval()
	val_unions = [esp for _ in range(cfg.MODEL_NUM_CLASSES)]
	val_insections = [0.0 for _ in range(cfg.MODEL_NUM_CLASSES)]
	val_loss = 0.0
	with torch.no_grad():
		for i ,data in tqdm.tqdm(enumerate(dataloader)):
			img, mask = data
			img = Variable(img).float().cuda()
			mask = Variable(mask).long().cuda()
			# img.cuda()
			# mask.cuda()
			output = net(img)
			loss = criterion(output, mask)
			compute_iou(output,mask,val_unions,val_insections)
			# print(loss.item())
			val_loss += loss.item()

		logger.add_scalars('loss', {'val':val_loss/len(dataloader)}, epoch)
		logger.add_scalars('bg', {'val':val_insections[0]/val_unions[0]}, epoch )
		logger.add_scalars('Bsmoke', {'val':val_insections[1]/val_unions[1]}, epoch)
		logger.add_scalars('Corn', {'val':val_insections[2]/val_unions[2]},epoch)
		logger.add_scalars('Brice', {'val':val_insections[3]/val_unions[3]}, epoch )
	print('loss:{:.6f}\tBsmoke:{:.6f}\tCorn:{:.6f}\tBrice:{:.6f}\tBG:{:.6f}'.format(
		val_loss/len(dataloader), val_insections[1] / val_unions[1], val_insections[2] / val_unions[2]
		, val_insections[3] / val_unions[3],val_insections[0] / val_unions[0]))

def compute_iou(output,mask,unions,insections,num = cfg.MODEL_NUM_CLASSES):
	# 0 is background
	B,C,H,W = output.shape
	output = torch.argmax(output,dim = 1)#.cpu().numpy()
	output = output.view(B, 1, H, W)
	outputs = torch.LongTensor(B, C, H, W).zero_().cuda()
	outputs = outputs.scatter_(1,output,1.)

	masks = torch.LongTensor(B, C, H, W).zero_().cuda()
	masks = masks.scatter_(1, mask, 1.)

	assert masks.shape == outputs.shape


	batch_insections = torch.sum(outputs * masks,dim = (0,2,3))
	batch_unions = torch.sum(outputs,dim=(0,2,3)) + torch.sum(masks,dim = (0,2,3)) - batch_insections
	for i in range(0,num):
		unions[i] += batch_unions[i].item()
		insections[i] += batch_insections[i].item()
	# for i in range(1,num):#class
	#
	# 	target_i = target[:, i].detach().cpu().numpy()
	# 	out_put_i = output[:, i].detach().cpu().numpy()
	# 	insection_i = target_i * out_put_i
	# 	union_i = target_i +  out_put_i - insection_i
	#
	# 	out_put_i = out_put_i.sum(axis = (1,2))
	# 	target_i = target_i.sum(axis = (1,2))
	# 	insection_i = insection_i.sum(axis = (1,2))
	# 	union_i = union_i.sum(axis = (1,2))
	# 	# print(out_put_i,target_i,insections,unions,sep='\n')
	# 	for j in range(n):
	# 		# if target_i[j] == 0:continue
	# 		# else:
	# 		# 	counts[i] = counts[i] + 1
	# 		# 	ious[i] += insections[j]/unions[j]
	# 		#另一种计算方式
	# 		unions[i] += union_i[j]
	# 		insections[i] += insection_i[j]


def adjust_lr(optimizer, epoch, max_epoch = cfg.TRAIN_EPOCHS):
	now_lr = cfg.TRAIN_LR * (1 - epoch/(max_epoch+1)) ** cfg.TRAIN_POWER
	optimizer.param_groups[0]['lr'] = now_lr
	# optimizer.param_groups[1]['lr'] = now_lr * 10
	return now_lr

def weights_init(m):
	if isinstance(m, nn.Conv2d):
		nn.init.xavier_normal_(m.weight.data)
		nn.init.xavier_normal_(m.bias.data)
	elif isinstance(m, nn.BatchNorm2d):
		nn.init.constant_(m.weight,1)
		nn.init.constant_(m.bias, 0)
	elif isinstance(m, nn.BatchNorm1d):
		nn.init.constant_(m.weight,1)
		nn.init.constant_(m.bias, 0)

def get_params(model, key):
	for m in model.named_modules():
		if key == '1x':
			if 'backbone' in m[0] and isinstance(m[1], nn.Conv2d):
				for p in m[1].parameters():
					yield p
		elif key == '10x':
			if 'backbone' not in m[0] and isinstance(m[1], nn.Conv2d):
				for p in m[1].parameters():
					yield p
if __name__ == '__main__':
	train_net()


