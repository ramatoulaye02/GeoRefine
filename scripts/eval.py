import yaml
from src.engine.evaluator import Evaluator


def load_cfg(path: str):
	with open(path, "r") as f:
		return yaml.safe_load(f)


def main():
	cfg = load_cfg("configs/default.yaml")
	evaluator = Evaluator(cfg)
	evaluator.run()


if __name__ == "__main__":
	main()
