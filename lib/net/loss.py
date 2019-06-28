
import torch
import torch.nn as nn
import torch.nn.functional as F
class MaskLoss(nn.Module):
	# bg:0.694139,Bsmoke:0.088105,Corn:0.072427,Brice:0.145329 样本数分布
	# weight bg:0.04303,Bsmoke:0.33902,Corn:0.41242,Brice:0.20553
	def __init__(self):
		super(MaskLoss, self).__init__()
		self.loss = None
		self.alpha = 0.25
		self.gamma = 2
		self.weight = torch.Tensor([0.04303,0.33902,0.41242,0.20553]).cuda()
	def forward(self,output, mask):
		#focal loss
		# output = F.softmax(output,dim=1)
		# probs,classes = torch.max(output,1)
		# mask.squeeze()
		# probs.squeeze()
		# classes.squeeze()

		pk = output * mask  # [B, C , H, W]
		pk = torch.clamp(pk, 1e-4, 1. - 1e-4).float()

		out = torch.pow((1. - pk), self.gamma) * torch.log(pk)

		batch_loss = -self.alpha * out.mean()
		# print batch_loss
		return batch_loss

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