from pathlib import Path
import argparse
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from rankrag.io import load_config
from rankrag.ranker.preprocessing import prepare_ranker_dataset


def prepare_ranker_dataset_main() -> None:
    parser = argparse.ArgumentParser(description="Precompute ranker features into tensor shards")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    manifest = prepare_ranker_dataset(load_config(args.config))
    print(json.dumps({"manifest": str(manifest)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    prepare_ranker_dataset_main()
