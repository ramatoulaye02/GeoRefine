import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.dataset import PairedEnhanceDataset
from src.models.unet import GeoUNet
from src.losses.perceptual import VGGPerceptualLoss
from src.losses.geo_losses import depth_normal_consistency
from src.utils.metrics import psnr
from src.utils.io import save_image_rgb, to_image_uint8


class Trainer:
	def __init__(self, cfg):
		self.cfg = cfg
		self.device = torch.device("cuda" if torch.cuda.is_available() and cfg["project"]["device"] == "cuda" else "cpu")

		train_ds = PairedEnhanceDataset(
			cfg["data"]["train_dir"],
			image_size=cfg["data"]["image_size"],
			use_depth=cfg["data"]["use_depth"],
			use_normals=cfg["data"]["use_normals"],
		)
		val_ds = PairedEnhanceDataset(
			cfg["data"]["val_dir"],
			image_size=cfg["data"]["image_size"],
			use_depth=cfg["data"]["use_depth"],
			use_normals=cfg["data"]["use_normals"],
		)

		self.train_loader = DataLoader(train_ds, batch_size=cfg["data"]["batch_size"], shuffle=True, num_workers=cfg["data"]["num_workers"])
		self.val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=0)

		in_ch = 3 + (1 if cfg["data"]["use_depth"] else 0) + (3 if cfg["data"]["use_normals"] else 0)
		self.model = GeoUNet(
			in_channels=in_ch,
			out_channels=3,
			base_channels=cfg["model"]["base_channels"],
			num_res_blocks=cfg["model"]["num_res_blocks"],
		).to(self.device)

		self.l1 = nn.L1Loss()
		self.perc = VGGPerceptualLoss().to(self.device)

		self.opt = torch.optim.AdamW(self.model.parameters(), lr=cfg["train"]["lr"], weight_decay=cfg["train"]["weight_decay"])

	def fit(self):
		epochs = self.cfg["train"]["epochs"]
		for ep in range(1, epochs + 1):
			self.model.train()
			pbar = tqdm(self.train_loader, desc=f"epoch {ep}/{epochs}")
			for step, batch in enumerate(pbar, 1):
				x = batch["x"].to(self.device)
				tgt = batch["tgt"].to(self.device)

				pred = self.model(x)

				loss = 0.0
				loss_l1 = self.l1(pred, tgt) * self.cfg["loss"]["l1_weight"]
				loss = loss + loss_l1

				loss_p = self.perc(pred, tgt) * self.cfg["loss"]["perceptual_weight"]
				loss = loss + loss_p

				# geo loss (only if normals exist)
				if self.cfg["data"]["use_depth"] and self.cfg["data"]["use_normals"]:
					depth = batch["depth"].to(self.device)  # (B,1,H,W)
					normals = batch["normals"].to(self.device)  # (B,3,H,W)
					loss_geo = depth_normal_consistency(depth, normals) * self.cfg["loss"]["geo_depth_normal_weight"]
					loss = loss + loss_geo
				else:
					loss_geo = torch.tensor(0.0)

				self.opt.zero_grad(set_to_none=True)
				loss.backward()
				self.opt.step()

				pbar.set_postfix({
					"l1": float(loss_l1.item()),
					"perc": float(loss_p.item()),
					"geo": float(loss_geo.item()) if isinstance(loss_geo, torch.Tensor) else float(loss_geo),
				})

			if ep % self.cfg["train"]["save_every"] == 0:
				ckpt_path = os.path.join(self.cfg["outputs"]["checkpoints"], f"model_ep{ep}.pt")
				torch.save(self.model.state_dict(), ckpt_path)
				self._save_val_sample(ep)

	@torch.no_grad()
	def _save_val_sample(self, ep: int):
		self.model.eval()
		batch = next(iter(self.val_loader))
		x = batch["x"].to(self.device)
		tgt = batch["tgt"].to(self.device)
		pred = self.model(x).clamp(0, 1)

		score = psnr(pred, tgt)
		out = to_image_uint8(pred.squeeze(0).cpu())
		path = os.path.join(self.cfg["outputs"]["samples"], f"val_ep{ep}_psnr{score:.2f}.png")
		save_image_rgb(out, path)
