import os
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.dataset import PairedEnhanceDataset
from src.models.unet import GeoUNet
from src.utils.metrics import psnr, ssim, LPIPSWrapper


class Evaluator:
	def __init__(self, cfg):
		self.cfg = cfg
		self.device = torch.device("cuda" if torch.cuda.is_available() and cfg["project"]["device"] == "cuda" else "cpu")

		val_ds = PairedEnhanceDataset(
			cfg["data"]["val_dir"],
			image_size=cfg["data"]["image_size"],
			use_depth=cfg["data"]["use_depth"],
			use_normals=cfg["data"]["use_normals"],
		)
		self.loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=0)

		in_ch = 3 + (1 if cfg["data"]["use_depth"] else 0) + (3 if cfg["data"]["use_normals"] else 0)
		self.model = GeoUNet(in_channels=in_ch, out_channels=3, base_channels=cfg["model"]["base_channels"], num_res_blocks=cfg["model"]["num_res_blocks"]).to(self.device)

		# grab latest checkpoint automatically
		ckpt_dir = cfg["outputs"]["checkpoints"]
		ckpts = sorted([os.path.join(ckpt_dir, f) for f in os.listdir(ckpt_dir) if f.endswith(".pt")])
		if not ckpts:
			raise RuntimeError(f"No checkpoints found in {ckpt_dir}")
		self.ckpt = ckpts[-1]
		self.model.load_state_dict(torch.load(self.ckpt, map_location=self.device))
		self.model.eval()

		self.lpips = LPIPSWrapper() if cfg["eval"]["compute_lpips"] else None

	@torch.no_grad()
	def run(self):
		psnrs, ssims, lpips_vals = [], [], []
		for batch in tqdm(self.loader, desc=f"eval ({os.path.basename(self.ckpt)})"):
			x = batch["x"].to(self.device)
			tgt = batch["tgt"].to(self.device)
			pred = self.model(x).clamp(0, 1)

			if self.cfg["eval"]["compute_psnr"]:
				psnrs.append(psnr(pred, tgt))
			if self.cfg["eval"]["compute_ssim"]:
				ssims.append(ssim(pred, tgt))
			if self.lpips is not None:
				lpips_vals.append(self.lpips(pred.cpu(), tgt.cpu()))

		def avg(xs):
			return sum(xs) / len(xs) if xs else None

		print("Checkpoint:", self.ckpt)
		print("PSNR:", avg(psnrs))
		print("SSIM:", avg(ssims))
		print("LPIPS:", avg(lpips_vals))
