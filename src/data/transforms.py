import torch
import torch.nn.functional as F
import random


def random_crop_resize(inp: torch.Tensor, tgt: torch.Tensor, size: int):
	# inp/tgt: (3,H,W)
	_, H, W = inp.shape
	if H < size or W < size:
		inp = F.interpolate(inp.unsqueeze(0), size=(max(H, size), max(W, size)), mode="bilinear", align_corners=False).squeeze(0)
		tgt = F.interpolate(tgt.unsqueeze(0), size=(max(H, size), max(W, size)), mode="bilinear", align_corners=False).squeeze(0)
		_, H, W = inp.shape

	top = random.randint(0, H - size)
	left = random.randint(0, W - size)
	inp = inp[:, top:top + size, left:left + size]
	tgt = tgt[:, top:top + size, left:left + size]
	return inp, tgt


def pad_to_multiple(x: torch.Tensor, multiple: int = 16):
	# x: (B,C,H,W)
	B, C, H, W = x.shape
	pad_h = (multiple - (H % multiple)) % multiple
	pad_w = (multiple - (W % multiple)) % multiple
	pad = (0, pad_w, 0, pad_h)
	x = F.pad(x, pad, mode="reflect")
	return x, pad


def unpad(x: torch.Tensor, pad):
	# pad: (left,right,top,bottom) but we used (0,pad_w,0,pad_h)
	_, pad_w, _, pad_h = pad
	if pad_h > 0:
		x = x[:, :, :-pad_h, :]
	if pad_w > 0:
		x = x[:, :, :, :-pad_w]
	return x
