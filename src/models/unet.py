import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
	def __init__(self, c):
		super().__init__()
		self.conv1 = nn.Conv2d(c, c, 3, padding=1)
		self.conv2 = nn.Conv2d(c, c, 3, padding=1)
		self.gn1 = nn.GroupNorm(8, c)
		self.gn2 = nn.GroupNorm(8, c)

	def forward(self, x):
		h = F.silu(self.gn1(self.conv1(x)))
		h = self.gn2(self.conv2(h))
		return F.silu(x + h)


class Down(nn.Module):
	def __init__(self, cin, cout):
		super().__init__()
		self.conv = nn.Conv2d(cin, cout, 4, stride=2, padding=1)

	def forward(self, x):
		return self.conv(x)


class Up(nn.Module):
	def __init__(self, cin, cout):
		super().__init__()
		self.conv = nn.Conv2d(cin, cout, 3, padding=1)

	def forward(self, x):
		x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
		return self.conv(x)


class GeoUNet(nn.Module):
	def __init__(self, in_channels=7, out_channels=3, base_channels=64, num_res_blocks=2):
		super().__init__()
		c = base_channels

		self.in_conv = nn.Conv2d(in_channels, c, 3, padding=1)

		self.d1 = Down(c, c * 2)
		self.d2 = Down(c * 2, c * 4)

		self.mid = nn.Sequential(*[ResBlock(c * 4) for _ in range(num_res_blocks)])

		self.u2 = Up(c * 4, c * 2)
		self.u1 = Up(c * 2, c)

		self.out = nn.Conv2d(c, out_channels, 3, padding=1)

	def forward(self, x):
		x0 = self.in_conv(x)
		x1 = self.d1(x0)
		x2 = self.d2(x1)

		m = self.mid(x2)

		y1 = self.u2(m)
		y0 = self.u1(y1)

		return torch.sigmoid(self.out(y0))
