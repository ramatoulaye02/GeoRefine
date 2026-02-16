import os
import cv2
import numpy as np
import torch


def ensure_dirs(paths):
	for p in paths:
		os.makedirs(p, exist_ok=True)


def load_image_rgb(path: str) -> np.ndarray:
	img = cv2.imread(path, cv2.IMREAD_COLOR)
	if img is None:
		raise FileNotFoundError(path)
	img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
	return img


def save_image_rgb(img_rgb_uint8: np.ndarray, path: str):
	img = cv2.cvtColor(img_rgb_uint8, cv2.COLOR_RGB2BGR)
	cv2.imwrite(path, img)


def to_tensor_01(img_rgb_uint8: np.ndarray) -> torch.Tensor:
	x = torch.from_numpy(img_rgb_uint8).float() / 255.0
	x = x.permute(2, 0, 1).contiguous()
	return x


def to_image_uint8(x_chw_01: torch.Tensor) -> np.ndarray:
	x = (x_chw_01.clamp(0, 1) * 255.0).round().byte()
	x = x.permute(1, 2, 0).cpu().numpy()
	return x
