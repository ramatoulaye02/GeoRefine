import torch
import torch.nn.functional as F
from skimage.metrics import structural_similarity as ssim_np
import numpy as np


def psnr(pred: torch.Tensor, target: torch.Tensor, eps=1e-8) -> float:
	mse = F.mse_loss(pred, target).item()
	if mse < eps:
		return 99.0
	return 10.0 * np.log10(1.0 / mse)


def ssim(pred: torch.Tensor, target: torch.Tensor) -> float:
	# pred/target: (1,3,H,W) in [0,1]
	p = pred.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
	t = target.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
	return float(ssim_np(t, p, channel_axis=2, data_range=1.0))


class LPIPSWrapper:
	def __init__(self, net="alex"):
		import lpips
		self.fn = lpips.LPIPS(net=net)

	@torch.no_grad()
	def __call__(self, pred: torch.Tensor, target: torch.Tensor) -> float:
		# lpips expects [-1,1]
		p = pred * 2 - 1
		t = target * 2 - 1
		v = self.fn(p, t).mean().item()
		return float(v)
