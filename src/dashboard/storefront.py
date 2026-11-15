"""Render the SAME storefront HTML in both preview and Streamlit.

The standalone preview is the source of truth for the UI; embedding it
prevents Streamlit's widget layout/theme from altering its appearance.
Only demo or CSV catalog data is presented. No fake model output is synthesized.
"""
from __future__ import annotations

import base64
from io import BytesIO
import html as html_lib
import json
import re
from pathlib import Path
from typing import Any

from PIL import Image, ImageOps
from src.dashboard.catalog import image_for, load_products
from src.dashboard.integration import load_search_reports, load_recommendation_metrics


def _img_data(path: Path) -> str:
    """Small JPG thumbnails prevent giant full-resolution CSV images in iframe HTML."""
    from src.dashboard.catalog import ASSET_DIR
    try:
        with Image.open(path) as photo:
            image = ImageOps.exif_transpose(photo).convert('RGB')
            image.thumbnail((460, 560), Image.Resampling.LANCZOS)
            buffer = BytesIO()
            image.save(buffer, format='JPEG', quality=78, optimize=True)
            raw = buffer.getvalue()
    except (OSError, ValueError):
        raw = (ASSET_DIR / 'no_image.jpg').read_bytes()
    return 'data:image/jpeg;base64,' + base64.b64encode(raw).decode('ascii')


def storefront_html(root: Path, *, embedded: bool = True) -> tuple[str, str]:
    """Return (HTML, data_source). Preview and Streamlit share identical CSS/DOM/JS."""
    root = Path(root)
    preview = (root / 'codisay_preview.html').read_text(encoding='utf-8')
    products, source = load_products(root)
    # The bundled preview uses the same 7 demo items; don't needlessly touch it.
    if source != 'demo':
        adapted = []
        for item in products[:48]:
            p = dict(item)
            p['image_url'] = _img_data(Path(image_for(item)))
            p['tags'] = list(item.get('tags') or [])
            p['rating'] = item.get('rating')
            p['price'] = item.get('price')
            p['badge'] = item.get('badge') or ''
            adapted.append(p)
        js = json.dumps(adapted, ensure_ascii=False).replace('</', '<\\/')
        preview, hits = re.subn(r'(const PRODUCTS=).*?(;\s*const grid=)',
                                 lambda m: m.group(1) + js + m.group(2), preview, count=1, flags=re.S)
        if hits != 1:
            raise ValueError('HTML 미리보기에서 상품 목록을 찾지 못했습니다.')
        # Correctly identify non-demo catalog data in the on-screen notice.
        preview = preview.replace('샘플 상품 7개 · 데모 검색','CSV 상품 · 키워드 미리보기')
        preview = preview.replace('샘플 상품 ${list.length}개 · 키워드 데모','CSV 상품 ${list.length}개 · 키워드 검색')
        preview = preview.replace('이 화면의 상품·가격·평점·추천은 모두 UI 샘플입니다. 실제 쇼핑몰 주문 기능이 아닙니다.',
                                  '상품 정보는 CSV에서 읽었습니다. 검색은 단순 키워드 검색이며 AI 모델 출력이 아닙니다.')
    # Populate KPIs only from the current actual report files, never demo values.
    reports = load_search_reports(root)
    rec = load_recommendation_metrics(root)
    def mrr(key: str) -> str:
        value = ((reports.get(key) or {}).get('metrics') or {}).get('mrr')
        return f'{float(value):.4f}' if isinstance(value, (int, float)) else '미제공'
    timings = ((reports.get('latency_pilot') or {}).get('timings') or {})
    latency = (timings.get('text') or {}).get('p95_ms')
    values = {
        'BM25 전체 MRR@10': mrr('bm25_full'),
        'CLIP 파일럿 MRR@10': mrr('clip_pilot'),
        '추천 HitRate@50': f"{rec['hit_rate_at_50']:.4f}" if rec and 'hit_rate_at_50' in rec else '미제공',
        '텍스트 검색 p95': f'{float(latency):.1f} ms' if isinstance(latency, (int, float)) else '미제공',
    }
    for label, value in values.items():
        pat = r'(<label>' + re.escape(label) + r'</label><strong>)[^<]*(</strong>)'
        preview, count = re.subn(pat, lambda m: m.group(1) + html_lib.escape(value) + m.group(2), preview, count=1)
        if count != 1:
            raise ValueError('대시보드 지표 레이아웃이 변경되었습니다: ' + label)

    if embedded:
        # Streamlit iframe sandbox blocks top navigation, but permits user-triggered popups.
        # Open native analysis pages in a new tab from the storefront sidebar.
        script = r"""<script>
(function(){
  for(const pair of [['dashboard','dashboard'], ['ab','experiments']]){
    const btn=document.querySelector('[data-page="'+pair[0]+'"]');
    if(!btn)continue;
    btn.addEventListener('click',function(event){
      event.stopImmediatePropagation();
      event.preventDefault();
      const link=document.createElement('a');
      link.href='?page='+pair[1];
      link.target='_blank';
      link.rel='noopener noreferrer';
      document.body.appendChild(link);link.click();link.remove();
    },true);
  }
})();
</script>"""
        preview=preview.replace('</body>', script+'</body>')
    return preview, source
