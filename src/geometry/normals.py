import torch
import torch.nn.functional as F


def normals_from_depth(depth: torch.Tensor, eps=1e-6) -> torch.Tensor:
	"""
	depth: (B,1,H,W) in [0,1] (relative)
	Returns normals approx (B,3,H,W).
	Simple screen-space normals from depth gradients.
	"""
	dzdx = depth[:, :, :, 2:] - depth[:, :, :, :-2]
	dzdy = depth[:, :, 2:, :] - depth[:, :, :-2, :]

	dzdx = F.pad(dzdx, (1, 1, 0, 0))
	dzdy = F.pad(dzdy, (0, 0, 1, 1))

	nx = -dzdx
	ny = -dzdy
	nz = torch.ones_like(depth)

	n = torch.cat([nx, ny, nz], dim=1)
	n = n / (torch.norm(n, dim=1, keepdim=True) + eps)
	return n
