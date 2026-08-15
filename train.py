from pathlib import Path
import argparse
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from rankrag.io import load_config
from rankrag.ranker.trainer import train_tensor_ranker


def train_main() -> None:
    parser = argparse.ArgumentParser(description="Train a ranker from precomputed tensor shards")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    if not config.get("ranker_dataset", {}).get("manifest"):
        raise ValueError("ranker_dataset.manifest is required; run prepare_ranker_dataset.py first")
    print(json.dumps(train_tensor_ranker(config), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    train_main()
