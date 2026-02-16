import os
import yaml
import torch
from src.utils.seed import set_seed
from src.engine.trainer import Trainer
from src.utils.io import ensure_dirs


def load_cfg(path: str):
	with open(path, "r") as f:
		return yaml.safe_load(f)


def main():
	cfg = load_cfg("configs/default.yaml")
	set_seed(cfg["project"]["seed"])

	ensure_dirs([
		cfg["outputs"]["dir"],
		cfg["outputs"]["checkpoints"],
		cfg["outputs"]["samples"],
	])

	trainer = Trainer(cfg)
	trainer.fit()


if __name__ == "__main__":
	main()
