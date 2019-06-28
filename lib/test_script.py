from lib.net.backbone import build_backbone
import torch
from torchsummary import summary

net = build_backbone('xception',pretrained=False)
#torch.save(net.state_dict(),'/home/yude/project/pretrained/netmodel.pth')

summary(net,(3,512,512))
