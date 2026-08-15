from __future__ import annotations

from collections import defaultdict
import json
import os
from pathlib import Path
import sqlite3
from typing import Any, Iterable, Iterator, Sequence

import numpy as np

from rankrag.data.paragraph_corpus import ParagraphCorpus
from rankrag.embedding import TextEmbedder


def normalize_entity(value: str) -> str:
    return " ".join(value.casefold().split())


def _chunks(values: Sequence[str], size: int = 800) -> Iterator[Sequence[str]]:
    for start in range(0, len(values), size):
        yield values[start : start + size]


def build_semantic_assets(
    corpus_path: str | Path,
    embeddings_path: str | Path,
    faiss_path: str | Path,
    embedder: TextEmbedder,
    batch_size: int = 256,
) -> dict[str, Any]:
    """Encode the corpus once and persist row-aligned embeddings and FAISS."""
    try:
        import faiss
    except ImportError as exc:
        raise RuntimeError("FAISS is required; install faiss-cpu or faiss-gpu before building retrieval assets") from exc

    corpus = ParagraphCorpus.load(corpus_path)
    embeddings_destination = Path(embeddings_path)
    faiss_destination = Path(faiss_path)
    embeddings_destination.parent.mkdir(parents=True, exist_ok=True)
    faiss_destination.parent.mkdir(parents=True, exist_ok=True)
    matrix = np.lib.format.open_memmap(
        embeddings_destination.with_suffix(embeddings_destination.suffix + ".tmp"),
        mode="w+",
        dtype=np.float32,
        shape=(len(corpus), embedder.dimension),
    )
    index = faiss.IndexFlatIP(embedder.dimension)
    for start in range(0, len(corpus), batch_size):
        texts = [f"{record.title}\n{record.text}" for record in corpus.records[start : start + batch_size]]
        vectors = np.asarray(embedder.encode(texts), dtype=np.float32)
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        vectors /= np.maximum(norms, 1e-12)
        matrix[start : start + len(vectors)] = vectors
        index.add(vectors)
        if start and start % (batch_size * 100) == 0:
            print(f"embedded={start}/{len(corpus)}", flush=True)
    matrix.flush()
    del matrix
    os.replace(
        embeddings_destination.with_suffix(embeddings_destination.suffix + ".tmp"),
        embeddings_destination,
    )
    temporary_faiss = faiss_destination.with_suffix(faiss_destination.suffix + ".tmp")
    faiss.write_index(index, str(temporary_faiss))
    os.replace(temporary_faiss, faiss_destination)
    return {"paragraph_count": len(corpus), "embedding_dim": embedder.dimension, "faiss_ntotal": int(index.ntotal)}


class SemanticParagraphIndex:
    def __init__(
        self,
        corpus: ParagraphCorpus,
        embeddings_path: str | Path,
        faiss_path: str | Path | None,
        backend: str = "faiss",
    ) -> None:
        self.corpus = corpus
        self.embeddings = np.load(embeddings_path, mmap_mode="r")
        if self.embeddings.shape[0] != len(corpus):
            raise ValueError("Corpus and embedding row counts do not match")
        self.backend = backend
        self.index = None
        if backend == "faiss":
            try:
                import faiss
            except ImportError as exc:
                raise RuntimeError("FAISS is required for query retrieval") from exc
            if faiss_path is None or not Path(faiss_path).exists():
                raise FileNotFoundError(f"FAISS index not found: {faiss_path}")
            self.index = faiss.read_index(str(faiss_path))
            if int(self.index.ntotal) != len(corpus):
                raise ValueError("Corpus and FAISS row counts do not match")
        elif backend != "numpy":
            raise ValueError(f"Unknown semantic index backend: {backend}")

    @property
    def dimension(self) -> int:
        return int(self.embeddings.shape[1])

    def search(self, query_vector: np.ndarray, top_k: int) -> list[tuple[str, float]]:
        top_k = min(max(0, top_k), len(self.corpus))
        if not top_k:
            return []
        vector = np.asarray(query_vector, dtype=np.float32).reshape(1, -1)
        vector /= max(float(np.linalg.norm(vector)), 1e-12)
        if self.backend == "faiss":
            scores, rows = self.index.search(vector, top_k)
            return [
                (self.corpus.records[int(row)].paragraph_id, float(score))
                for row, score in zip(rows[0], scores[0], strict=True)
                if int(row) >= 0
            ]
        scores = np.asarray(self.embeddings @ vector[0])
        rows = np.argsort(-scores, kind="stable")[:top_k]
        return [(self.corpus.records[int(row)].paragraph_id, float(scores[int(row)])) for row in rows]

    def scores(self, query_vector: np.ndarray, paragraph_ids: Sequence[str]) -> dict[str, float]:
        if not paragraph_ids:
            return {}
        rows = [self.corpus.id_to_row[pid] for pid in paragraph_ids]
        vector = np.asarray(query_vector, dtype=np.float32)
        vector /= max(float(np.linalg.norm(vector)), 1e-12)
        values = np.asarray(self.embeddings[rows] @ vector)
        return {pid: float(score) for pid, score in zip(paragraph_ids, values, strict=True)}

    def close(self) -> None:
        mmap = getattr(self.embeddings, "_mmap", None)
        if mmap is not None:
            mmap.close()


def build_global_graph_index(
    corpus_path: str | Path,
    extractions_path: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    """Build disk-backed paragraph/entity mappings and relation adjacency."""
    corpus = ParagraphCorpus.load(corpus_path)
    paragraphs_by_title: dict[str, list[str]] = defaultdict(list)
    for record in corpus.records:
        paragraphs_by_title[normalize_entity(record.title)].append(record.paragraph_id)

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    if temporary.exists():
        temporary.unlink()
    connection = sqlite3.connect(temporary)
    connection.executescript(
        """
        PRAGMA journal_mode=OFF;
        PRAGMA synchronous=OFF;
        CREATE TABLE entities (
            entity_id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            entity_type TEXT NOT NULL,
            description TEXT NOT NULL
        );
        CREATE TABLE paragraph_to_entities (
            paragraph_id TEXT NOT NULL,
            entity_id TEXT NOT NULL,
            PRIMARY KEY (paragraph_id, entity_id)
        );
        CREATE TABLE entity_to_paragraphs (
            entity_id TEXT NOT NULL,
            paragraph_id TEXT NOT NULL,
            PRIMARY KEY (entity_id, paragraph_id)
        );
        CREATE TABLE entity_relation_adjacency (
            source_entity_id TEXT NOT NULL,
            target_entity_id TEXT NOT NULL,
            relation TEXT NOT NULL,
            description TEXT NOT NULL,
            paragraph_id TEXT NOT NULL,
            PRIMARY KEY (source_entity_id, target_entity_id, relation, paragraph_id)
        );
        """
    )
    records = mapped_records = 0
    with Path(extractions_path).open("r", encoding="utf-8") as handle:
        for line in handle:
            records += 1
            value = json.loads(line)
            entities = list(value.get("entities", []))
            title = str(value.get("title") or (entities[0].get("entity_name", "") if entities else "")).strip()
            extraction_id = str(value.get("id", ""))
            paragraph_ids = [extraction_id] if extraction_id in corpus.id_to_row else paragraphs_by_title.get(normalize_entity(title), [])
            if not paragraph_ids:
                continue
            mapped_records += 1
            entity_ids: set[str] = set()
            for entity in entities:
                name = str(entity.get("entity_name", "")).strip()
                entity_id = normalize_entity(name)
                if not entity_id:
                    continue
                entity_ids.add(entity_id)
                connection.execute(
                    "INSERT OR IGNORE INTO entities VALUES (?, ?, ?, ?)",
                    (entity_id, name, str(entity.get("entity_type", "entity")), str(entity.get("description", ""))),
                )
                for pid in paragraph_ids:
                    connection.execute("INSERT OR IGNORE INTO paragraph_to_entities VALUES (?, ?)", (pid, entity_id))
                    connection.execute("INSERT OR IGNORE INTO entity_to_paragraphs VALUES (?, ?)", (entity_id, pid))
            for relation in value.get("relationships", []):
                source_name = str(relation.get("src_id", "")).strip()
                if source_name == extraction_id:
                    source_name = title
                target_name = str(relation.get("tgt_id", "")).strip()
                source_id = normalize_entity(source_name)
                target_id = normalize_entity(target_name)
                if not source_id or not target_id or source_id not in entity_ids or target_id not in entity_ids:
                    continue
                relation_name = str(relation.get("keywords", "related_to")) or "related_to"
                for pid in paragraph_ids:
                    connection.execute(
                        "INSERT OR IGNORE INTO entity_relation_adjacency VALUES (?, ?, ?, ?, ?)",
                        (source_id, target_id, relation_name, str(relation.get("description", "")), pid),
                    )
                    connection.execute(
                        "INSERT OR IGNORE INTO entity_relation_adjacency VALUES (?, ?, ?, ?, ?)",
                        (target_id, source_id, relation_name, str(relation.get("description", "")), pid),
                    )
            if records % 10000 == 0:
                connection.commit()
                print(f"kg_records={records} mapped={mapped_records}", flush=True)
    connection.executescript(
        """
        CREATE INDEX idx_p2e_paragraph ON paragraph_to_entities(paragraph_id);
        CREATE INDEX idx_e2p_entity ON entity_to_paragraphs(entity_id);
        CREATE INDEX idx_rel_source ON entity_relation_adjacency(source_entity_id);
        """
    )
    counts = {
        "kg_records": records,
        "mapped_kg_records": mapped_records,
        "paragraph_entity_edges": connection.execute("SELECT COUNT(*) FROM paragraph_to_entities").fetchone()[0],
        "entity_paragraph_edges": connection.execute("SELECT COUNT(*) FROM entity_to_paragraphs").fetchone()[0],
        "entity_relation_edges": connection.execute("SELECT COUNT(*) FROM entity_relation_adjacency").fetchone()[0],
    }
    connection.commit()
    connection.close()
    os.replace(temporary, destination)
    return counts


class GlobalGraphIndex:
    """Read-only access to the three global KG mappings."""

    def __init__(self, path: str | Path) -> None:
        uri = f"file:{Path(path).resolve().as_posix()}?mode=ro"
        self.connection = sqlite3.connect(uri, uri=True)

    def entities_for_paragraphs(self, paragraph_ids: Sequence[str]) -> dict[str, list[str]]:
        result: dict[str, list[str]] = defaultdict(list)
        for chunk in _chunks(list(dict.fromkeys(paragraph_ids))):
            placeholders = ",".join("?" for _ in chunk)
            rows = self.connection.execute(
                f"SELECT paragraph_id, entity_id FROM paragraph_to_entities WHERE paragraph_id IN ({placeholders}) ORDER BY paragraph_id, entity_id",
                tuple(chunk),
            )
            for paragraph_id_value, entity_id in rows:
                result[paragraph_id_value].append(entity_id)
        return dict(result)

    def paragraphs_for_entities(self, entity_ids: Sequence[str], per_entity_limit: int) -> dict[str, list[str]]:
        result: dict[str, list[str]] = defaultdict(list)
        for chunk in _chunks(list(dict.fromkeys(entity_ids))):
            placeholders = ",".join("?" for _ in chunk)
            rows = self.connection.execute(
                f"SELECT entity_id, paragraph_id FROM entity_to_paragraphs WHERE entity_id IN ({placeholders}) ORDER BY entity_id, paragraph_id",
                tuple(chunk),
            )
            for entity_id, paragraph_id_value in rows:
                if len(result[entity_id]) < per_entity_limit:
                    result[entity_id].append(paragraph_id_value)
        return dict(result)

    def adjacent_entities(
        self,
        entity_ids: Sequence[str],
        per_entity_limit: int,
    ) -> dict[str, list[dict[str, str]]]:
        result: dict[str, list[dict[str, str]]] = defaultdict(list)
        for chunk in _chunks(list(dict.fromkeys(entity_ids))):
            placeholders = ",".join("?" for _ in chunk)
            rows = self.connection.execute(
                f"SELECT source_entity_id, target_entity_id, relation, description FROM entity_relation_adjacency WHERE source_entity_id IN ({placeholders}) ORDER BY source_entity_id, target_entity_id",
                tuple(chunk),
            )
            for source, target, relation, description in rows:
                if len(result[source]) < per_entity_limit:
                    result[source].append({"target": target, "relation": relation, "description": description})
        return dict(result)

    def entity_metadata(self, entity_ids: Iterable[str]) -> dict[str, dict[str, str]]:
        values = list(dict.fromkeys(entity_ids))
        result: dict[str, dict[str, str]] = {}
        for chunk in _chunks(values):
            placeholders = ",".join("?" for _ in chunk)
            rows = self.connection.execute(
                f"SELECT entity_id, name, entity_type, description FROM entities WHERE entity_id IN ({placeholders})",
                tuple(chunk),
            )
            for entity_id, name, entity_type, description in rows:
                result[entity_id] = {"name": name, "entity_type": entity_type, "description": description}
        return result

    def close(self) -> None:
        self.connection.close()
