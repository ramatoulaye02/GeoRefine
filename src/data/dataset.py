import os
import glob
import torch
from torch.utils.data import Dataset
from src.utils.io import load_image_rgb, to_tensor_01
from src.geometry.depth_midas import estimate_depth_stub
from src.geometry.normals import normals_from_depth
from src.data.transforms import random_crop_resize


class PairedEnhanceDataset(Dataset):
	"""
	Expects:
	  train_dir/
		input/  (degraded renders)
		target/ (clean/high-quality renders)
	Filenames should match 1:1.
	"""
	def __init__(self, root, image_size=256, use_depth=True, use_normals=True):
		self.inp_dir = os.path.join(root, "input")
		self.tgt_dir = os.path.join(root, "target")
		self.paths = sorted(glob.glob(os.path.join(self.inp_dir, "*.*")))
		self.image_size = image_size
		self.use_depth = use_depth
		self.use_normals = use_normals

	def __len__(self):
		return len(self.paths)

	def __getitem__(self, idx):
		inp_path = self.paths[idx]
		name = os.path.basename(inp_path)
		tgt_path = os.path.join(self.tgt_dir, name)

		inp = to_tensor_01(load_image_rgb(inp_path))
		tgt = to_tensor_01(load_image_rgb(tgt_path))

		inp, tgt = random_crop_resize(inp, tgt, self.image_size)

		# build conditioned input
		x = inp.unsqueeze(0)  # (1,3,H,W)
		conds = []
		depth = None
		normals = None

		if self.use_depth:
			depth = estimate_depth_stub(x)  # (1,1,H,W)
			conds.append(depth)
		if self.use_normals and depth is not None:
			normals = normals_from_depth(depth)  # (1,3,H,W)
			conds.append(normals)

		x = torch.cat([x] + conds, dim=1) if conds else x
		x = x.squeeze(0)

		return {
			"x": x,          # (C,H,W) conditioned input
			"inp": inp,      # (3,H,W) raw input
			"tgt": tgt,      # (3,H,W) target
			"depth": depth.squeeze(0) if depth is not None else None,
			"normals": normals.squeeze(0) if normals is not None else None,
			"name": name,
		}
