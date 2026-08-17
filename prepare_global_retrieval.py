from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from rankrag.data.paragraph_corpus import build_paragraph_corpus, inspect_paragraph_corpus
from rankrag.embedding import create_embedder
from rankrag.graphrag.global_assets import (
    build_global_graph_index,
    build_semantic_assets,
    inspect_global_graph_index,
    inspect_semantic_assets,
)
from rankrag.io import load_config, write_json


def _paths(config: dict) -> dict[str, Path]:
    global_config = config.get("global_retrieval", {})
    asset_dir = Path(global_config.get("asset_dir", "outputs/global_retrieval"))
    return {
        "asset_dir": asset_dir,
        "corpus": Path(global_config.get("corpus_path", asset_dir / "corpus.jsonl")),
        "embeddings": Path(global_config.get("embeddings_path", asset_dir / "paragraph_embeddings.npy")),
        "faiss": Path(global_config.get("faiss_index_path", asset_dir / "paragraphs.faiss")),
        "graph": Path(global_config.get("graph_index_path", asset_dir / "global_graph.sqlite")),
        "manifest": Path(global_config.get("manifest_path", asset_dir / "manifest.json")),
    }


def _write_manifest(paths: dict[str, Path], manifest: dict) -> None:
    manifest["assets"] = {key: str(value) for key, value in paths.items() if key != "asset_dir"}
    write_json(paths["manifest"], manifest)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build cached global HotpotQA retrieval assets")
    parser.add_argument("--config", required=True)
    parser.add_argument("--stage", choices=["all", "corpus", "embeddings", "graph"], default="all")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    global_config = config.get("global_retrieval", {})
    if not global_config.get("enabled", False):
        raise ValueError("global_retrieval.enabled must be true")
    paths = _paths(config)
    paths["asset_dir"].mkdir(parents=True, exist_ok=True)
    manifest = json.loads(paths["manifest"].read_text(encoding="utf-8")) if paths["manifest"].exists() else {"schema_version": 1}

    if args.stage in ("all", "corpus"):
        inputs = global_config.get("corpus_inputs", [])
        if not inputs:
            raise ValueError("global_retrieval.corpus_inputs must contain at least one paragraph JSON file")
        if args.force or not paths["corpus"].exists():
            manifest["corpus"] = build_paragraph_corpus(inputs, paths["corpus"])
        else:
            print(f"Reusing corpus: {paths['corpus']}")
            if "corpus" not in manifest:
                manifest["corpus"] = inspect_paragraph_corpus(inputs, paths["corpus"])
                print("Recovered missing corpus metadata in asset manifest")
        _write_manifest(paths, manifest)

    if args.stage in ("all", "embeddings"):
        if not paths["corpus"].exists():
            raise FileNotFoundError(f"Build corpus first: {paths['corpus']}")
        if args.force or not (paths["embeddings"].exists() and paths["faiss"].exists()):
            manifest["semantic_index"] = build_semantic_assets(
                paths["corpus"],
                paths["embeddings"],
                paths["faiss"],
                create_embedder(config.get("embedding", {})),
                batch_size=int(global_config.get("embedding_batch_size", 256)),
            )
            manifest["embedding"] = config.get("embedding", {})
        else:
            print(f"Reusing embeddings and FAISS: {paths['embeddings']}, {paths['faiss']}")
            if "semantic_index" not in manifest:
                manifest["semantic_index"] = inspect_semantic_assets(paths["embeddings"], paths["faiss"])
                print("Recovered missing semantic index metadata in asset manifest")
        manifest["embedding"] = config.get("embedding", {})
        _write_manifest(paths, manifest)

    if args.stage in ("all", "graph"):
        if not paths["corpus"].exists():
            raise FileNotFoundError(f"Build corpus first: {paths['corpus']}")
        extraction_path = Path(config.get("graph", {}).get("extractions_path", ""))
        if not extraction_path.exists():
            raise FileNotFoundError(f"KG extractions not found: {extraction_path}")
        if args.force or not paths["graph"].exists():
            manifest["global_graph"] = build_global_graph_index(paths["corpus"], extraction_path, paths["graph"])
        else:
            print(f"Reusing global graph: {paths['graph']}")
            if "global_graph" not in manifest:
                manifest["global_graph"] = inspect_global_graph_index(paths["graph"])
                print("Recovered available global graph metadata in asset manifest")
        _write_manifest(paths, manifest)

    _write_manifest(paths, manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
