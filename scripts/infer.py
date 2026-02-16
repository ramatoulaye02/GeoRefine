import argparse
import yaml
import torch
from src.models.unet import GeoUNet
from src.utils.io import load_image_rgb, save_image_rgb
from src.geometry.depth_midas import estimate_depth_stub
from src.geometry.normals import normals_from_depth
from src.data.transforms import pad_to_multiple, unpad
from src.utils.io import to_tensor_01, to_image_uint8


def load_cfg(path: str):
	with open(path, "r") as f:
		return yaml.safe_load(f)


@torch.no_grad()
def main():
	ap = argparse.ArgumentParser()
	ap.add_argument("--ckpt", required=True)
	ap.add_argument("--inp", required=True)
	ap.add_argument("--out", required=True)
	args = ap.parse_args()

	cfg = load_cfg("configs/default.yaml")
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	model = GeoUNet(
		in_channels=3 + (1 if cfg["data"]["use_depth"] else 0) + (3 if cfg["data"]["use_normals"] else 0),
		out_channels=3,
		base_channels=cfg["model"]["base_channels"],
		num_res_blocks=cfg["model"]["num_res_blocks"],
	).to(device)
	model.load_state_dict(torch.load(args.ckpt, map_location=device))
	model.eval()

	img = load_image_rgb(args.inp)
	x = to_tensor_01(img).unsqueeze(0)  # (1,3,H,W)

	# geometry conditioning (stub depth, replace with real MiDaS later)
	conds = []
	if cfg["data"]["use_depth"]:
		d = estimate_depth_stub(x)  # (1,1,H,W)
		conds.append(d)
		if cfg["data"]["use_normals"]:
			n = normals_from_depth(d)  # (1,3,H,W)
			conds.append(n)
	elif cfg["data"]["use_normals"]:
		# if no depth, normals are meaningless; keep simple
		pass

	if conds:
		x = torch.cat([x] + conds, dim=1)

	x, pad = pad_to_multiple(x, multiple=16)
	x = x.to(device)

	y = model(x).clamp(0, 1)
	y = unpad(y, pad)

	out_img = to_image_uint8(y.squeeze(0).cpu())
	save_image_rgb(out_img, args.out)


if __name__ == "__main__":
	main()
