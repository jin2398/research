"""No-model/no-download setup check for the two team ZIP bundles."""
from __future__ import annotations
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parent
INDEX=ROOT/'artifacts/search/clip_hnsw_full'
SOURCE=ROOT/'src/search'
missing=[str(p.relative_to(ROOT)) for p in [
    SOURCE/'clip_faiss.py', SOURCE/'query_text.py', SOURCE/'text_fusion.py',
    INDEX/'products.index', INDEX/'products.jsonl', INDEX/'manifest.json'] if not p.is_file()]
print('\nRE:SEARCH 검색 연결 상태')
for folder in ['src/search','artifacts/search/clip_hnsw_full']:
    print('폴더:',folder)
if missing:
    for item in missing: print('  [누락]',item)
else:
    manifest=json.loads((INDEX/'manifest.json').read_text(encoding='utf-8'))
    count=sum(1 for line in (INDEX/'products.jsonl').open(encoding='utf-8') if line.strip())
    print('인덱스 상품 수:',manifest.get('indexed_products'))
    print('메타데이터 행 수:',count)
    if count != manifest.get('indexed_products'):
        missing.append('인덱스의 메타데이터 행 수 불일치')
    if manifest.get('model_name') != 'openai/clip-vit-base-patch32':
        missing.append('CLIP 모델명 불일치')
print('상태:', '누락/오류 있음' if missing else '검색 ZIP 설치 경로와 메타데이터 일치')
print('참고: CLIP 가중치 로딩과 브라우저 검색은 여기서 검사하지 않습니다.')
if missing:raise SystemExit(1)
