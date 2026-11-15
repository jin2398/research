"""Lightweight catalog compatibility layer for the supplied CLIP/search update.

The included full index can be searched without any CSVs: this module is mainly
for the types used by the original search code and future full-index rebuilding.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import csv

@dataclass(slots=True)
class CatalogItem:
    product_id: str
    name: str
    price: int
    category_small: str
    color: str
    pattern: str
    image_path: Path

@dataclass(slots=True)
class CatalogStats:
    product_rows: int = 0
    feature_rows: int = 0
    missing_features: int = 0
    missing_images: int = 0
    included: int = 0


def load_catalog(products_csv: Path, features_csv: Path, image_root: Path,
                 limit: int | None = None) -> tuple[list[CatalogItem], CatalogStats]:
    """Read a typical product/features table; not needed for the prebuilt index.

    The team must provide the original CSVs and images before rerunning an index
    build or offline evaluation. These CSV columns may need adaptation if their
    upstream schema differs.
    """
    products_csv, features_csv, image_root = map(Path, (products_csv, features_csv, image_root))
    if not products_csv.is_file() or not features_csv.is_file():
        raise FileNotFoundError('인덱스 재생성에는 data/products.csv 및 data/product_features.csv가 필요합니다.')
    with features_csv.open(encoding='utf-8-sig', newline='') as file:
        feats = {str(row.get('product_id') or row.get('article_id') or ''): row for row in csv.DictReader(file)}
    result: list[CatalogItem] = []
    stats = CatalogStats(feature_rows=len(feats))
    with products_csv.open(encoding='utf-8-sig', newline='') as file:
        for row in csv.DictReader(file):
            stats.product_rows += 1
            pid = str(row.get('product_id') or row.get('article_id') or '').strip()
            feat = feats.get(pid)
            if not pid or feat is None:
                stats.missing_features += 1
                continue
            path = image_root / (pid + '.jpg')
            if not path.is_file():
                stats.missing_images += 1
                continue
            raw = row.get('price') or feat.get('price') or '0'
            try: price = int(float(raw))
            except (ValueError, TypeError): price = 0
            result.append(CatalogItem(pid,
                str(row.get('name') or row.get('product_name') or feat.get('name') or pid), price,
                str(feat.get('category_small') or feat.get('product_type_name') or ''),
                str(feat.get('color') or ''), str(feat.get('pattern') or ''),path))
            if limit is not None and len(result) >= limit:
                break
    stats.included = len(result)
    return result, stats
