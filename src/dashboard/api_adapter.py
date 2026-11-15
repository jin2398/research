"""Optional HTTP integration: team API schema must be confirmed before production use."""
from __future__ import annotations

import os
from typing import Any
from urllib.parse import urlsplit

import requests


def get_api_base() -> str:
    base = os.getenv('CODISAY_API_BASE_URL', '').strip().rstrip('/')
    parts = urlsplit(base)
    return base if parts.scheme in ('http', 'https') and parts.netloc else ''


def parse_results(data: Any) -> list[dict]:
    if isinstance(data, dict):
        data = next((data[key] for key in ('results', 'products', 'recommendations', 'items')
                     if isinstance(data.get(key), list)), None)
    if not isinstance(data, list):
        raise ValueError('API 응답에 results/products/recommendations/items 배열이 없습니다.')
    products=[]
    for obj in data[:100]:
        if not isinstance(obj, dict):
            continue
        identifier = str(obj.get('product_id', obj.get('id', ''))).strip()
        if not identifier:
            continue
        raw_price = obj.get('price')
        try: price = int(float(raw_price)) if raw_price is not None else None
        except (ValueError, TypeError): price = None
        products.append({'id':identifier, 'name':str(obj.get('name') or obj.get('title') or identifier),
                         'price':price, 'category':str(obj.get('category') or '기타'),
                         'style':str(obj.get('style') or ''), 'color':str(obj.get('color') or ''),
                         'image':None, 'badge':'API', 'rating':None,
                         'reason':str(obj.get('reason') or ''), 'tags':[]})
    return products


def search_api(query: str, mode: str = 'text', top_k: int = 20) -> list[dict]:
    base = get_api_base()
    if not base:
        raise ValueError('CODISAY_API_BASE_URL이 설정되지 않았습니다.')
    # Image multipart API contract is not available in the uploaded patch.
    if mode != 'text':
        raise ValueError('이미지·하이브리드 검색 API의 업로드 규격은 팀원과 확인 후 연결해야 합니다.')
    try:
        response = requests.post(f'{base}/api/search', json={'query_text':query,'top_k':top_k},timeout=8)
        response.raise_for_status()
        return parse_results(response.json())
    except (requests.RequestException, ValueError) as exc:
        raise ValueError(f'검색 API 요청 실패: {exc}') from exc


def recommend_api(user_id: str, k: int = 20) -> list[dict]:
    base = get_api_base()
    if not base:
        raise ValueError('CODISAY_API_BASE_URL이 설정되지 않았습니다.')
    try:
        response = requests.get(f'{base}/api/recommend', params={'user_id':user_id, 'top_n':k}, timeout=8)
        response.raise_for_status()
        return parse_results(response.json())
    except (requests.RequestException, ValueError) as exc:
        raise ValueError(f'추천 API 요청 실패: {exc}') from exc
