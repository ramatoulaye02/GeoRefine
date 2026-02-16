import torch
import torch.nn.functional as F


def depth_normal_consistency(depth: torch.Tensor, normals: torch.Tensor) -> torch.Tensor:
	"""
	Simple proxy: align depth gradients with normal xy components.
	depth: (B,1,H,W)
	normals: (B,3,H,W)
	"""
	dzdx = depth[:, :, :, 1:] - depth[:, :, :, :-1]
	dzdy = depth[:, :, 1:, :] - depth[:, :, :-1, :]

	nx = normals[:, 0:1]
	ny = normals[:, 1:2]

	nx = nx[:, :, :, 1:]  # match shapes roughly
	ny = ny[:, :, 1:, :]

	loss_x = F.l1_loss(dzdx, -nx)
	loss_y = F.l1_loss(dzdy, -ny)
	return loss_x + loss_y
