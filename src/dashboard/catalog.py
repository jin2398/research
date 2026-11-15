"""Fashion storefront catalog and honest demo search helpers.

The bundled products are UI fixtures only: no images, prices, ratings, or
recommendation explanations here are inferred from real model outputs.
"""
from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import Any

ASSET_DIR = Path(__file__).resolve().parent / "assets"

DEMO_PRODUCTS = [
    {"id": "demo-001", "name": "클래식 싱글 롱코트", "price": 189000, "category": "아우터", "color": "베이지", "style": "미니멀", "rating": 4.8, "image": "product_1.jpg", "badge": "추천", "tags": ["베이지", "코트", "가을", "데일리", "오피스", "코디"]},
    {"id": "demo-002", "name": "레더 블루종 자켓", "price": 129000, "category": "아우터", "color": "브라운", "style": "캐주얼", "rating": 4.7, "image": "product_2.jpg", "badge": "신상품", "tags": ["가죽", "브라운", "자켓", "블루종", "가을", "코디"]},
    {"id": "demo-003", "name": "케이블 니트 스웨터", "price": 59000, "category": "상의", "color": "아이보리", "style": "미니멀", "rating": 4.8, "image": "product_3.jpg", "badge": "유사 스타일", "tags": ["니트", "스웨터", "아이보리", "겨울", "미니멀", "코디"]},
    {"id": "demo-004", "name": "와이드 핀턱 슬랙스", "price": 69000, "category": "하의", "color": "브라운", "style": "미니멀", "rating": 4.7, "image": "product_4.jpg", "badge": "추천", "tags": ["바지", "슬랙스", "와이드", "오피스", "데일리", "코디"]},
    {"id": "demo-005", "name": "플라워 미디 원피스", "price": 89000, "category": "원피스", "color": "블랙", "style": "페미닌", "rating": 4.6, "image": "product_5.jpg", "badge": "신상품", "tags": ["원피스", "플라워", "데이트", "페미닌", "봄", "코디"]},
    {"id": "demo-006", "name": "클래식 스니커즈", "price": 79000, "category": "신발", "color": "화이트", "style": "캐주얼", "rating": 4.8, "image": "product_6.jpg", "badge": "유사 스타일", "tags": ["스니커즈", "신발", "운동화", "데일리", "캐주얼", "코디"]},
    {"id": "demo-007", "name": "미니 숄더백", "price": 49000, "category": "가방", "color": "브라운", "style": "미니멀", "rating": 4.7, "image": "product_7.jpg", "badge": "추천", "tags": ["가방", "숄더백", "미니백", "브라운", "데일리", "코디"]},
]


def demo_products() -> list[dict[str, Any]]:
    return [dict(p) for p in DEMO_PRODUCTS]


def load_products(root: Path, limit: int = 1500) -> tuple[list[dict[str, Any]], str]:
    """Read agreed-on data/products.csv contract or return *labeled* UI fixtures."""
    csv_path = root / 'data/products.csv'
    if not csv_path.is_file():
        return demo_products(), 'demo'
    products = []
    with csv_path.open(encoding='utf-8-sig', newline='') as fp:
        rows = csv.DictReader(fp)
        fields = set(rows.fieldnames or [])
        # Clear error rather than silently treating an incompatible team's CSV as valid.
        if not {'product_id', 'name', 'category'}.issubset(fields):
            raise ValueError('data/products.csv에는 product_id, name, category 열이 필요합니다.')
        for row in rows:
            if len(products) >= limit:
                break
            ident = str(row.get('product_id') or '').strip()
            name = str(row.get('name') or '').strip()
            if not ident or not name:
                continue
            raw_price = str(row.get('price', '')).replace(',', '').strip()
            try:
                price = int(float(raw_price)) if raw_price else None
                if price is not None and price < 0:
                    price = None
            except (TypeError, ValueError):
                price = None
            image_path = str(row.get('image_path') or '').strip()
            # Only local paths rooted in the project are accepted for CSV image paths.
            image = None
            if image_path and not Path(image_path).is_absolute():
                candidate = (root / image_path).resolve()
                if candidate.is_relative_to(root.resolve()) and candidate.is_file():
                    image = str(candidate)
            products.append({'id': ident, 'name': name, 'category': str(row.get('category') or '기타'),
                             'color': str(row.get('color') or ''), 'style': str(row.get('style') or ''),
                             'price': price, 'image': image, 'badge': '', 'rating': None,
                             'tags': [str(row.get('category') or ''), str(row.get('color') or ''), str(row.get('style') or '')]})
    return products, 'csv'


def image_for(product: dict[str, Any]) -> Path | str:
    val = product.get('image')
    if val and Path(str(val)).is_file():
        return str(val)
    if val and (ASSET_DIR / str(val)).is_file():
        return str(ASSET_DIR / str(val))
    return str(ASSET_DIR / 'no_image.jpg')


def match_score(product: dict[str, Any], query: str) -> int:
    tokens = [tok for tok in re.findall(r'[가-힣a-zA-Z0-9]+', query.casefold()) if len(tok) > 1]
    if not tokens:
        return 1
    name = str(product.get('name') or '').casefold()
    fields = ' '.join(str(x).casefold() for x in [product.get('category'), product.get('style'),
                                                  product.get('color'), *(product.get('tags') or [])])
    return sum(3 if t in name else (1 if t in fields else 0) for t in tokens)


def filter_products(products: list[dict[str, Any]], query: str = '', category: str = '전체',
                    order: str = '추천순') -> list[dict[str, Any]]:
    ranked = [(match_score(p, query), p) for p in products
              if category == '전체' or p.get('category') == category]
    ranked = [(score, p) for score, p in ranked if score > 0]
    if order == '낮은 가격순':
        ranked.sort(key=lambda v: (v[1].get('price') is None, v[1].get('price') or 0, -v[0]))
    elif order == '높은 가격순':
        ranked.sort(key=lambda v: (v[1].get('price') is None, -(v[1].get('price') or 0), -v[0]))
    else:
        ranked.sort(key=lambda v: (-v[0], v[1].get('id', '')))
    return [p for _, p in ranked]


def suggest_products(products: list[dict[str, Any]], style: str, count: int = 4) -> list[dict[str, Any]]:
    # Explicitly simple rule-based demo; not a trained model or user history.
    priority = [p for p in products if p.get('style') == style]
    remaining = [p for p in products if p not in priority]
    return (priority + remaining)[:count]
