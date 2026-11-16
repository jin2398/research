"""Select text rankers on validation, then measure on the fixed search test."""

from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import asdict
from pathlib import Path
from time import perf_counter

from src.evaluation.metrics import evaluate_rankings

from .bm25 import BM25Index
from .catalog import load_catalog
from .clip_faiss import ClipFaissSearch, SearchConfig
from .query_text import normalize_query_text
from .relevance import AttributeRelevance, search_queries
from .text_fusion import AttributeMatcher, FusionSettings, fuse_rankings


class TextRankers:
    """Reuse a loaded CLIP index and a lexical index for text experiments."""

    def __init__(self, config: SearchConfig) -> None:
        self.engine = ClipFaissSearch(config)
        products, _ = load_catalog(
            config.products_csv, config.features_csv, config.image_root
        )
        if [product.product_id for product in products] != self.ids:
            raise ValueError(
                "Experiment requires the full saved index and its matching catalog"
            )
        self.bm25 = BM25Index(products)
        self.attributes = AttributeMatcher(products)
        self.judgments = AttributeRelevance(products)
        self.normalize = config.normalize_korean_text

    @property
    def ids(self) -> list[str]:
        return [row["product_id"] for row in self.engine.products]

    def lexical_query(self, query: str) -> str:
        return normalize_query_text(query) if self.normalize else query

    def rank(
        self, query: str, method: str, settings: FusionSettings | None = None
    ) -> list[str]:
        lexical = self.lexical_query(query)
        if method == "bm25":
            return self.bm25.search(lexical, 10)
        if method == "clip":
            return [row["product_id"] for row in self.engine.search(query)["results"]]
        if method != "fusion" or settings is None:
            raise ValueError("Choose bm25, clip, or fusion with settings")
        bm25 = self.bm25.search(lexical, settings.depth)
        clip = (
            [
                row["product_id"]
                for row in self.engine.search(query, top_k=settings.depth)["results"]
            ]
            if settings.clip_weight
            else []
        )
        penalties = (
            self.attributes.penalties(lexical, bm25 + clip)
            if settings.match_attributes
            else None
        )
        return fuse_rankings(bm25, clip, settings, penalties)

    def prepare(self, texts: list[str], depths: set[int], batch_size: int) -> dict:
        ids = self.ids
        bm25 = {
            text: self.bm25.search(self.lexical_query(text), max(depths))
            for text in texts
        }
        clip: dict[int, dict[str, list[str]]] = {depth: {} for depth in depths}
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            vectors = self.engine.encoder.encode_texts(batch)
            for depth in sorted(depths):
                _, indices = self.engine.index.search(vectors, depth)
                clip[depth].update(
                    (text, [ids[int(i)] for i in numbers if i >= 0])
                    for text, numbers in zip(batch, indices)
                )
        penalties = {
            text: self.attributes.penalties(
                self.lexical_query(text),
                list(
                    dict.fromkeys(
                        bm25[text]
                        + [pid for depth in depths for pid in clip[depth][text]]
                    )
                ),
            )
            for text in texts
        }
        return {"bm25": bm25, "clip": clip, "penalties": penalties}


def evaluate_split(
    rankers: TextRankers,
    split: Path,
    candidates: dict[str, list[FusionSettings]],
    batch_size: int,
) -> dict:
    events = list(search_queries(split))
    texts = list(dict.fromkeys(row["search_query"].strip() for row in events))
    depths = {
        10,
        *(setting.depth for variants in candidates.values() for setting in variants),
    }
    print(
        f"Retrieving {len(texts)} unique texts from {split.name} ({len(events)} events)",
        flush=True,
    )
    prepared = rankers.prepare(texts, depths, batch_size)
    judgment_cache: dict[tuple[str, ...], set[str]] = {}
    rows = []
    for event in events:
        key = tuple(
            event.get(name, "")
            for name in ("query_category_small", "query_color", "query_pattern")
        )
        if key not in judgment_cache:
            judgment_cache[key] = rankers.judgments.relevant_ids(event)
        rows.append((event["search_query"].strip(), judgment_cache[key]))

    def metrics(ranked: dict[str, list[str]]) -> dict:
        return asdict(
            evaluate_rankings((ranked[text], relevance) for text, relevance in rows)
        )

    baselines = {
        "bm25": metrics({text: prepared["bm25"][text][:10] for text in texts}),
        "clip": metrics(prepared["clip"][10]),
    }
    results = {}
    for family, variants in candidates.items():
        scores = []
        for setting in variants:
            ranked = {
                text: fuse_rankings(
                    prepared["bm25"][text],
                    prepared["clip"][setting.depth][text],
                    setting,
                    prepared["penalties"][text],
                )
                for text in texts
            }
            scores.append({"settings": asdict(setting), "metrics": metrics(ranked)})
        scores.sort(
            key=lambda entry: (entry["metrics"]["ndcg_at_10"], entry["metrics"]["mrr"]),
            reverse=True,
        )
        results[family] = scores
        print(f"{split.name} {family}: {scores[0]}", flush=True)
    return {
        "split": str(split),
        "events": len(rows),
        "unique_queries": len(texts),
        "baselines": baselines,
        "rankers": results,
    }


def benchmark(
    rankers: TextRankers, test_csv: Path, selected: dict[str, FusionSettings]
) -> dict:
    queries = list(
        dict.fromkeys(row["search_query"].strip() for row in search_queries(test_csv))
    )
    queries = random.Random(42).sample(queries, 110)
    methods = [
        ("bm25", "bm25", None),
        ("clip", "clip", None),
        *((name, "fusion", settings) for name, settings in selected.items()),
    ]
    timings = {name: [] for name, _, _ in methods}
    for i, query in enumerate(queries):
        offset = i % len(methods)
        for name, method, settings in methods[offset:] + methods[:offset]:
            start = perf_counter()
            rankers.rank(query, method, settings)
            if i >= 10:
                timings[name].append((perf_counter() - start) * 1000)
    return {
        name: {
            "p50_ms": round(sorted(samples)[49], 3),
            "p95_ms": round(sorted(samples)[math.ceil(0.95 * len(samples)) - 1], 3),
        }
        for name, samples in timings.items()
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("config/search_full.yaml"))
    parser.add_argument(
        "--validation", type=Path, default=Path("data/splits/valid.csv")
    )
    parser.add_argument("--test", type=Path, default=Path("data/splits/test.csv"))
    parser.add_argument(
        "--output", type=Path, default=Path("docs/search_text_fusion.json")
    )
    args = parser.parse_args()
    if args.validation.resolve() == args.test.resolve():
        parser.error("Validation and test must be separate")
    config = SearchConfig.from_yaml(args.config)
    print("Loading the full CLIP index and BM25 catalog", flush=True)
    rankers = TextRankers(config)
    candidates = {"rank_fusion": [], "attribute_bm25": [], "attribute_fusion": []}
    for depth in (50, 100):
        candidates["attribute_bm25"].append(FusionSettings(depth, 0, 10, True))
        for constant in (10, 60):
            for weight in (0.05, 0.15, 0.3, 0.5):
                candidates["rank_fusion"].append(
                    FusionSettings(depth, weight, constant, False)
                )
                candidates["attribute_fusion"].append(
                    FusionSettings(depth, weight, constant, True)
                )
    validation = evaluate_split(rankers, args.validation, candidates, config.batch_size)
    selected = {
        name: FusionSettings(**scores[0]["settings"])
        for name, scores in validation["rankers"].items()
    }
    print("Validation selection locked; evaluating on the test split", flush=True)
    test = evaluate_split(
        rankers,
        args.test,
        {name: [value] for name, value in selected.items()},
        config.batch_size,
    )
    print("Measuring warmed text rankers on 100 unique test queries", flush=True)
    timings = benchmark(rankers, args.test, selected)
    report = {
        "catalog_size": rankers.engine.index.ntotal,
        "judgment": "same category_small and every explicitly queried color/pattern",
        "query_inputs": "search_query only; no event labels or target product IDs in the ranker",
        "selection": "Each family: highest validation NDCG@10, then MRR@10; test does not tune settings.",
        "validation": validation,
        "test": test,
        "latency": {
            "scope": "warmed text ranking to top 10 IDs, excludes HTTP and startup",
            "device": config.device,
            "warmup": 10,
            "samples": 100,
            "seed": 42,
            "torch_threads": rankers.engine.encoder.torch.get_num_threads(),
            "timings": timings,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        json.dumps({"baselines": test["baselines"], "latency": timings}, indent=2),
        flush=True,
    )


if __name__ == "__main__":
    main()
