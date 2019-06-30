
import torch
import torch.nn as nn
import torch.nn.functional as F
class MaskLoss(nn.Module):
	# bg:0.694139,Bsmoke:0.088105,Corn:0.072427,Brice:0.145329 样本数分布
	# weight bg:0.04303,Bsmoke:0.33902,Corn:0.41242,Brice:0.20553
	def __init__(self):
		super(MaskLoss, self).__init__()
		self.loss = None
		self.alpha = 0.75
		self.gamma = 2
		self.weight = [1-0.694139,1-0.088105,1-0.072427,1-0.145329]
	def forward(self,output, mask):
		#focal loss
		# output = F.softmax(output,dim=1)
		# probs,classes = torch.max(output,1)
		# mask.squeeze()
		# probs.squeeze()
		# classes.squeeze()

		output = torch.sigmoid(output)
		#positive
		pos_pk = output * mask  # [B, C , H, W]
		pos_pk = torch.clamp(pos_pk, 1e-4, 1. - 1e-4).float()
		pos_loss = torch.pow((1. - pos_pk), self.gamma) * torch.log(pos_pk)
		#negative
		neg_pk = output * (1-mask)  # [B, C , H, W]
		neg_pk = torch.clamp(neg_pk, 1e-4, 1. - 1e-4).float()
		neg_loss = torch.pow(neg_pk, self.gamma) * torch.log(1 - neg_pk)
		for i,weight in enumerate(self.weight):
			pos_loss[:,i,:,:] *= weight
			neg_loss[:,i,:,:] *= (1 - weight)

		#正负样本比1:3
		batch_loss = -(self.alpha) * pos_loss.mean() - (1 - self.alpha) * neg_loss.mean()
		# print batch_loss
		return batch_loss

		# #BCE
		# loss = nn.CrossEntropyLoss()(output,mask.long())
		# return loss

#
# def focal_loss(Preds, Labels):
# 	B, C, H, W = Preds.size()
# 	Pred = torch.argmax(Preds, dim=1)  # [B, H, W]
# 	Label = torch.FloatTensor(B, C, H, W).zero_().cuda()
# 	Labels = Labels.view(B, 1, H, W)
# 	Label = Label.scatter_(1, Labels, 1.)
#
# 	pk = Preds * Label  # [B, H, W]
# 	pk = torch.clamp(pk, 1e-4, 1. - 1e-4).float()
#
# 	out = torch.pow((1. - pk), self.gamma) * torch.log(pk)
# 	batch_loss = -self.alpha * out.mean()
# 	# print batch_loss
# 	return batch_loss