import torch
import torch.nn as nn
from torchvision.models import vgg16, VGG16_Weights


class VGGPerceptualLoss(nn.Module):
	def __init__(self, layer_ids=(3, 8, 15, 22)):
		super().__init__()
		vgg = vgg16(weights=VGG16_Weights.DEFAULT).features.eval()
		for p in vgg.parameters():
			p.requires_grad = False
		self.vgg = vgg
		self.layer_ids = set(layer_ids)

	def forward(self, pred, target):
		# pred/target: (B,3,H,W) in [0,1]
		loss = 0.0
		x = pred
		y = target
		for i, layer in enumerate(self.vgg):
			x = layer(x)
			y = layer(y)
			if i in self.layer_ids:
				loss = loss + (x - y).abs().mean()
		return loss
