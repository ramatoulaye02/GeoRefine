import torch
import torch.nn.functional as F


def estimate_depth_stub(x_rgb: torch.Tensor) -> torch.Tensor:
	"""
	x_rgb: (B,3,H,W) in [0,1]
	Returns a fake "depth-like" map just to make the pipeline run.
	Replace with MiDaS later.
	"""
	# luminance as a stand-in depth proxy
	r, g, b = x_rgb[:, 0:1], x_rgb[:, 1:2], x_rgb[:, 2:3]
	d = 0.299 * r + 0.587 * g + 0.114 * b
	# smooth a bit
	d = F.avg_pool2d(d, kernel_size=7, stride=1, padding=3)
	return d
