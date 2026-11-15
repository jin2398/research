"""CLIP image index and text/image/hybrid FAISS search.

Build a small functional slice first::

    python -m src.search.clip_faiss build --limit 1000
    python -m src.search.clip_faiss query --text "black boots"

The full build omits ``--limit``. Image embeddings are computed offline;
query encoding and ANN lookup are the only online operations.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from .catalog import CatalogItem, load_catalog


@dataclass(frozen=True, slots=True)
class SearchConfig:
    products_csv: Path
    features_csv: Path
    image_root: Path
    artifacts_dir: Path
    model_cache_dir: Path
    model_name: str
    model_local_files_only: bool
    device: str
    batch_size: int
    checkpoint_every_batches: int
    hnsw_m: int
    ef_construction: int
    ef_search: int
    hybrid_text_weight: float
    normalize_korean_text: bool = False

    @classmethod
    def from_yaml(cls, path: Path) -> "SearchConfig":
        import yaml

        values = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(values, dict):
            raise ValueError(f"Expected a YAML mapping in {path}")
        root = os.environ.get("SEARCH_IMAGE_ROOT") or values["image_root"]
        config = cls(
            products_csv=Path(values["products_csv"]),
            features_csv=Path(values["features_csv"]),
            image_root=Path(root),
            artifacts_dir=Path(values["artifacts_dir"]),
            model_cache_dir=Path(values["model_cache_dir"]),
            model_name=str(values["model_name"]),
            model_local_files_only=bool(values["model_local_files_only"]),
            device=str(values["device"]),
            batch_size=int(values["batch_size"]),
            checkpoint_every_batches=int(values["checkpoint_every_batches"]),
            hnsw_m=int(values["hnsw_m"]),
            ef_construction=int(values["ef_construction"]),
            ef_search=int(values["ef_search"]),
            hybrid_text_weight=float(values["hybrid_text_weight"]),
            normalize_korean_text=bool(values.get("normalize_korean_text", False)),
        )
        if min(config.batch_size, config.checkpoint_every_batches,
               config.hnsw_m) < 1:
            raise ValueError("batch_size, checkpoint interval and hnsw_m must be positive")
        if not 0 <= config.hybrid_text_weight <= 1:
            raise ValueError("hybrid_text_weight must be between 0 and 1")
        return config


def _feature_tensor(result: Any) -> Any:
    """Accept the tensor output of Transformers 4 and structured newer APIs."""
    if hasattr(result, "detach"):
        return result
    if hasattr(result, "pooler_output") and result.pooler_output is not None:
        return result.pooler_output
    raise TypeError(f"Unexpected CLIP feature output: {type(result)!r}")


class ClipEncoder:
    def __init__(self, model_name: str, device: str = "cpu",
                 cache_dir: Path | None = None,
                 local_files_only: bool = True,
                 normalize_korean_text: bool = False) -> None:
        if cache_dir is not None:
            if not cache_dir.is_dir():
                cache_dir.mkdir(parents=True, exist_ok=True)
            os.environ.setdefault("HF_HOME", str(cache_dir.parent / "hf_home"))
            os.environ.setdefault("HF_HUB_CACHE", str(cache_dir))
        if local_files_only:
            # AutoProcessor may load nested tokenizer files without forwarding
            # local_files_only. Set the Hub process mode before its first import.
            os.environ["HF_HUB_OFFLINE"] = "1"
        import numpy as np
        import torch
        from transformers import AutoProcessor, CLIPModel

        self.np = np
        self.torch = torch
        self.device = torch.device(device)
        self.processor = AutoProcessor.from_pretrained(
            model_name, cache_dir=str(cache_dir) if cache_dir else None,
            local_files_only=local_files_only, use_fast=False,
        )
        self.model = CLIPModel.from_pretrained(
            model_name, cache_dir=str(cache_dir) if cache_dir else None,
            local_files_only=local_files_only,
        ).to(self.device).eval()
        self.dimension = int(self.model.config.projection_dim)
        self.normalize_korean_text = normalize_korean_text

    def _as_vectors(self, result: Any) -> Any:
        tensor = _feature_tensor(result)
        vectors = tensor.detach().float().cpu().numpy().astype("float32", copy=False)
        norms = self.np.linalg.norm(vectors, axis=1, keepdims=True)
        if self.np.any(norms == 0):
            raise ValueError("CLIP returned a zero-length embedding")
        return self.np.ascontiguousarray(vectors / norms)

    def encode_texts(self, texts: list[str], *, normalize: bool | None = None) -> Any:
        if not texts:
            raise ValueError("texts must not be empty")
        should_normalize = self.normalize_korean_text if normalize is None else normalize
        if should_normalize:
            from .query_text import normalize_query_text
            texts = [normalize_query_text(text) for text in texts]
        values = self.processor(text=texts, padding=True, truncation=True,
                                return_tensors="pt")
        values = {key: value.to(self.device) for key, value in values.items()}
        with self.torch.inference_mode():
            result = self.model.get_text_features(**values)
        return self._as_vectors(result)

    def encode_images(self, paths: list[Path]) -> Any:
        if not paths:
            raise ValueError("paths must not be empty")
        from PIL import Image

        with ExitStack() as stack:
            images = [stack.enter_context(Image.open(path)).convert("RGB")
                      for path in paths]
            values = self.processor(images=images, return_tensors="pt")
            values = {key: value.to(self.device) for key, value in values.items()}
            with self.torch.inference_mode():
                result = self.model.get_image_features(**values)
        return self._as_vectors(result)


def _batches(items: list[CatalogItem], size: int) -> Iterable[list[CatalogItem]]:
    for start in range(0, len(items), size):
        yield items[start:start + size]


def _check_index_disk_space(folder: Path, count: int, dimension: int,
                            hnsw_m: int) -> None:
    # A new index is written beside the saved checkpoint before atomic replace.
    # Allow for vector data, HNSW links, metadata overhead and OS headroom.
    required = count * (dimension * 4 + hnsw_m * 8 + 128) + 256 * 1024 * 1024
    available = shutil.disk_usage(folder).free
    if available < required:
        raise OSError(f"Insufficient space for index checkpoint: need {required:,} "
                      f"free bytes, have {available:,}; the previous checkpoint is retained")


def build_index(config: SearchConfig, *, limit: int | None = None,
                resume: bool = False) -> dict[str, Any]:
    import faiss

    products, stats = load_catalog(config.products_csv, config.features_csv,
                                   config.image_root, limit=limit)
    if not products:
        raise ValueError("No indexable products")
    encoder = ClipEncoder(config.model_name, config.device,
                          config.model_cache_dir,
                          config.model_local_files_only)
    config.artifacts_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = config.artifacts_dir / "products.jsonl"
    metadata_partial = config.artifacts_dir / "products.partial.jsonl"
    checkpoint_path = config.artifacts_dir / "products.partial.index"
    checkpoint_tmp = config.artifacts_dir / "products.partial.index.tmp"
    if resume:
        if not checkpoint_path.is_file() or not metadata_partial.is_file():
            raise FileNotFoundError("No index checkpoint to resume")
        index = faiss.read_index(str(checkpoint_path))
        if index.d != encoder.dimension or index.ntotal > len(products):
            raise ValueError("Checkpoint and requested catalog are incompatible")
        retained: list[str] = []
        with metadata_partial.open(encoding="utf-8") as handle:
            for position in range(index.ntotal):
                line = handle.readline()
                if not line:
                    raise ValueError("Checkpoint metadata is shorter than the index")
                if json.loads(line)["product_id"] != products[position].product_id:
                    raise ValueError("Checkpoint belongs to a different product catalog")
                retained.append(line)
        # A previous run may have written metadata after the last saved index.
        # Discard those lines so index positions and metadata remain aligned.
        metadata_partial.write_text("".join(retained), encoding="utf-8")
        start_at = index.ntotal
    else:
        checkpoint_path.unlink(missing_ok=True)
        checkpoint_tmp.unlink(missing_ok=True)
        metadata_partial.write_text("", encoding="utf-8")
        index = faiss.IndexHNSWFlat(encoder.dimension, config.hnsw_m,
                                   faiss.METRIC_INNER_PRODUCT)
        start_at = 0
    index.hnsw.efConstruction = config.ef_construction
    index.hnsw.efSearch = config.ef_search

    started = time.perf_counter()
    with metadata_partial.open("a", encoding="utf-8", newline="\n") as handle:
        for batch_number, batch in enumerate(
            _batches(products[start_at:], config.batch_size), 1
        ):
            vectors = encoder.encode_images([item.image_path for item in batch])
            index.add(vectors)
            for item in batch:
                handle.write(json.dumps({
                    "product_id": item.product_id,
                    "name": item.name,
                    "price": item.price,
                    "category_small": item.category_small,
                    "color": item.color,
                    "pattern": item.pattern,
                }, ensure_ascii=False) + "\n")
            if batch_number % config.checkpoint_every_batches == 0:
                handle.flush()
                _check_index_disk_space(config.artifacts_dir, index.ntotal,
                                        encoder.dimension, config.hnsw_m)
                faiss.write_index(index, str(checkpoint_tmp))
                os.replace(checkpoint_tmp, checkpoint_path)
                print(f"Indexed {index.ntotal}/{len(products)} products", flush=True)
    if index.ntotal != len(products):
        raise RuntimeError("Index and product metadata counts differ")

    index_path = config.artifacts_dir / "products.index"
    index_tmp = config.artifacts_dir / "products.index.tmp"
    _check_index_disk_space(config.artifacts_dir, index.ntotal,
                            encoder.dimension, config.hnsw_m)
    faiss.write_index(index, str(index_tmp))
    os.replace(index_tmp, index_path)
    os.replace(metadata_partial, metadata_path)
    checkpoint_path.unlink(missing_ok=True)
    manifest = {
        "index_type": "HNSWFlat_IP",
        "model_name": config.model_name,
        "dimension": encoder.dimension,
        "indexed_products": index.ntotal,
        "full_eligible_catalog": (
            stats.product_rows - stats.missing_features - stats.missing_images
        ),
        "partial_build": limit is not None,
        "hnsw_m": config.hnsw_m,
        "ef_construction": config.ef_construction,
        "ef_search": config.ef_search,
        "build_seconds": round(time.perf_counter() - started, 3),
        "catalog_stats": {
            "product_rows": stats.product_rows,
            "feature_rows": stats.feature_rows,
            "missing_features": stats.missing_features,
            "missing_images": stats.missing_images,
            "included": stats.included,
        },
    }
    manifest_path = config.artifacts_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2),
                             encoding="utf-8")
    return manifest


class ClipFaissSearch:
    """Adapter callable by the serving teammate for POST /api/search."""

    def __init__(self, config: SearchConfig) -> None:
        import faiss

        self.config = config
        folder = config.artifacts_dir
        manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
        if manifest["model_name"] != config.model_name:
            raise ValueError("Index and query encoder use different CLIP models")
        self.index = faiss.read_index(str(folder / "products.index"))
        self.index.hnsw.efSearch = config.ef_search
        with (folder / "products.jsonl").open(encoding="utf-8") as handle:
            self.products = [json.loads(line) for line in handle]
        if self.index.ntotal != len(self.products):
            raise ValueError("Index and product metadata counts differ")
        self.encoder = ClipEncoder(config.model_name, config.device,
                                   config.model_cache_dir,
                                   config.model_local_files_only,
                                   normalize_korean_text=config.normalize_korean_text)
        if self.index.d != self.encoder.dimension:
            raise ValueError("Index and model embedding dimensions differ")

    def search(
        self,
        query_text: str | None = None,
        query_image_path: str | None = None,
        top_k: int = 10,
    ) -> dict[str, Any]:
        import numpy as np

        started = time.perf_counter()
        text_query = (query_text or "").strip()
        if not text_query and not query_image_path:
            raise ValueError("query_text or query_image_path is required")
        if not 1 <= top_k <= 1000:
            raise ValueError("top_k must be in [1, 1000]")

        text_vector = (self.encoder.encode_texts([text_query])
                       if text_query else None)
        image_vector = (self.encoder.encode_images([Path(query_image_path)])
                        if query_image_path else None)
        if text_vector is not None and image_vector is not None:
            kind = "hybrid"
            weight = self.config.hybrid_text_weight
            query_vector = weight * text_vector + (1 - weight) * image_vector
            norm = np.linalg.norm(query_vector, axis=1, keepdims=True)
            if np.any(norm == 0):
                raise ValueError("Hybrid query has zero-length embedding")
            query_vector = np.ascontiguousarray(query_vector / norm,
                                                dtype="float32")
        elif text_vector is not None:
            kind, query_vector = "text", text_vector
        else:
            kind, query_vector = "image", image_vector

        scores, indices = self.index.search(query_vector, min(top_k, self.index.ntotal))
        results = []
        for score, number in zip(scores[0], indices[0]):
            if number < 0:
                continue
            product = self.products[int(number)]
            results.append({
                "product_id": product["product_id"],
                "name": product["name"],
                "score": max(-1.0, min(1.0, float(score))),
                "price": int(product["price"]),
            })
        return {
            "search_type": kind,
            "results": results,
            "latency_ms": round((time.perf_counter() - started) * 1000, 3),
            "total_count": len(results),
        }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("config/search.yaml"))
    actions = parser.add_subparsers(dest="action", required=True)
    build = actions.add_parser("build", help="Encode product images and save HNSW")
    build.add_argument("--limit", type=int,
                       help="Use the first N eligible items for a smoke run")
    build.add_argument("--allow-model-download", action="store_true",
                       help="Download CLIP weights if not cached locally")
    build.add_argument("--resume", action="store_true",
                       help="Continue from the last saved partial index")
    query = actions.add_parser("query", help="Search the saved index")
    query.add_argument("--text")
    query.add_argument("--image", help="Local query image path")
    query.add_argument("--top-k", type=int, default=10)
    args = parser.parse_args()
    config = SearchConfig.from_yaml(args.config)
    if args.action == "build":
        if args.allow_model_download:
            from dataclasses import replace
            config = replace(config, model_local_files_only=False)
        result = build_index(config, limit=args.limit, resume=args.resume)
    else:
        result = ClipFaissSearch(config).search(args.text, args.image, args.top_k)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
