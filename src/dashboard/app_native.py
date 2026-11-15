"""CodiSay fashion storefront + preserved team analytics.

Run at project root:
    python -m streamlit run src/dashboard/app.py

This is a UI-first demo. No fake model evaluation metrics are generated.
"""
from __future__ import annotations

import html
import sys
from pathlib import Path
from typing import Any

import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.dashboard.analytics_views import render_analytics
from src.dashboard.api_adapter import get_api_base, recommend_api, search_api
from src.dashboard.catalog import filter_products, image_for, load_products, suggest_products
from src.dashboard.integration import load_recommendation_metrics, load_search_reports, load_session_report
from src.dashboard.style import stylesheet

st.set_page_config(page_title='RE:SEARCH | AI 패션 검색·추천', page_icon='🧥', layout='wide',
                   initial_sidebar_state='expanded')
st.markdown(stylesheet(), unsafe_allow_html=True)

for key, initial in [('cs_nav', '홈'), ('cs_wishlist', set()), ('cs_search_query', ''),
                      ('cs_last_query', ''), ('cs_last_mode', '텍스트 검색'),
                      ('cs_search_submitted', False), ('cs_category', '전체'), ('cs_order', '추천순')]:
    if key not in st.session_state:
        st.session_state[key] = initial


def safe(x: Any) -> str:
    return html.escape(str(x))


def price_text(price: int | None) -> str:
    return f'₩ {price:,}' if price is not None else '가격 미제공'


def section(title: str, subtitle: str = ''):
    st.markdown(f'<div class="cs-section"><h3>{safe(title)}</h3><p>{safe(subtitle)}</p></div>',
                unsafe_allow_html=True)


def render_header(page: str):
    st.markdown(f'''<div class="cs-header">
        <div><span class="cs-header-brand">RE:SEARCH</span><span class="cs-header-sub">AI로 더 스마트한 패션 쇼핑</span></div>
        <div class="cs-header-side">{safe(page)} · UI v1.0</div>
    </div>''', unsafe_allow_html=True)


def render_card(product: dict, source: str, key: str):
    with st.container(border=True):
        st.image(image_for(product), use_container_width=True)
        badge = product.get('badge')
        if badge:
            st.markdown(f'<span class="cs-demo-chip">{safe(badge)}</span>', unsafe_allow_html=True)
        st.markdown(f'<div class="cs-product-title" title="{safe(product.get("name", ""))}">{safe(product.get("name", "상품"))}</div>',unsafe_allow_html=True)
        st.markdown(f'<div class="cs-product-meta">{safe(price_text(product.get("price")))}</div>', unsafe_allow_html=True)
        parts = [str(product.get('category') or '')]
        if product.get('rating') is not None and source == 'demo':
            parts.append(f'★ {product["rating"]:.1f} · 데모')
        st.markdown(f'<div class="cs-product-detail">{safe(" · ".join(parts))}</div>',unsafe_allow_html=True)
        in_wishlist = product['id'] in st.session_state.cs_wishlist
        if st.button('♥ 찜 해제' if in_wishlist else '♡ 찜하기', key=f'like_{key}_{product["id"]}',use_container_width=True):
            new = set(st.session_state.cs_wishlist)
            if in_wishlist: new.remove(product['id'])
            else: new.add(product['id'])
            st.session_state.cs_wishlist = new
            st.rerun()
        with st.expander('상품 정보'):
            st.write(f'상품 ID: `{product["id"]}`')
            st.write(f'카테고리: {product.get("category", "미제공")}')
            if product.get('reason'):
                st.write(f'추천 사유: {product["reason"]}')
            if source == 'demo':
                st.caption('화면 구현용 샘플 상품으로 실제 판매 정보가 아닙니다.')


def render_grid(products: list[dict], source: str, key='home', ncols=4, max_items=12):
    if not products:
        st.info('조건에 일치하는 상품이 없습니다. 다른 검색어 또는 카테고리를 선택해 주세요.')
        return
    selected = products[:max_items]
    for pos in range(0,len(selected),ncols):
        columns=st.columns(ncols,gap='small')
        for i,product in enumerate(selected[pos:pos+ncols]):
            with columns[i]:
                render_card(product, source, key)


def sidebar() -> str:
    st.sidebar.markdown('<div class="cs-brand-side">♧ RE:SEARCH</div>',unsafe_allow_html=True)
    nav = st.sidebar.radio('메뉴', ['홈', '멀티모달 검색', '개인화 추천', '대시보드', '실험관리'],
                           key='cs_nav', label_visibility='collapsed')
    st.sidebar.divider()
    base=get_api_base()
    st.sidebar.caption('연동 상태')
    if base:
        st.sidebar.success('외부 API 주소 설정됨')
        st.sidebar.caption('API 규격 확인 후 실제 요청을 사용하세요.')
        st.sidebar.toggle('실제 모델 API 사용', key='cs_use_real_api',value=False)
    else:
        st.sidebar.info('UI 데모 모드 · API 미연결')
        st.session_state.cs_use_real_api=False
    st.sidebar.markdown('''<div class="cs-side-bottom">AI로 찾는 나만의 스타일<br>RE:SEARCH<br><small>이미지 · 텍스트 검색 · 취향 추천</small></div>''',unsafe_allow_html=True)
    st.sidebar.caption('© RE:SEARCH · 학습용 서비스 시제품')
    return nav


def search_form(key: str) -> tuple[str,str,Any,bool]:
    # A separate form key per screen prevents unintended cross-page form state.
    with st.form(f'{key}_search_form',border=True):
        st.markdown('<div class="cs-panel-title">🔎 내 스타일 찾기</div>', unsafe_allow_html=True)
        mode=st.radio('검색 방식', ['텍스트 검색','이미지 검색','하이브리드 검색'],
                      horizontal=True,key=f'{key}_mode')
        col1,col2=st.columns([3.2,1.5], gap='medium')
        with col1:
            query=st.text_input('검색어', value=st.session_state.cs_search_query,
                                placeholder='예: 가을에 입기 좋은 베이지 코트',key=f'{key}_query')
        with col2:
            uploaded=st.file_uploader('이미지 업로드',type=['jpg','jpeg','png','webp'],key=f'{key}_image')
        submit=st.form_submit_button('스타일 검색하기',type='primary',use_container_width=True)
    if submit:
        st.session_state.cs_last_query=query.strip()
        st.session_state.cs_last_mode=mode
        st.session_state.cs_search_submitted=True
        st.session_state.cs_search_query=query
    return query,mode,uploaded,submit


def search_result(products: list[dict], source: str, query: str, mode: str, uploaded: Any, submit: bool):
    data = products
    effective_query=query if submit else (st.session_state.cs_last_query if st.session_state.cs_search_submitted else '')
    if st.session_state.cs_search_submitted:
        mode=st.session_state.cs_last_mode
        if mode in ('이미지 검색', '하이브리드 검색'):
            if uploaded is None:
                st.warning('이미지 검색을 위해 이미지를 업로드해 주세요.')
                return
            st.image(uploaded,caption='업로드한 이미지 · AI 유사도 분석 전',width=190)
            st.info('이미지는 정상 업로드됐어요. CLIP 모델 API 연결 전이므로 실제 이미지 유사도 검색은 수행하지 않습니다. 아래는 화면 시연용 카탈로그입니다.')
        elif not effective_query:
            st.info('검색어를 입력해 주세요.')
            return
    category_options=['전체']+sorted(set(str(p.get('category') or '기타') for p in data))
    if st.session_state.cs_category not in category_options: st.session_state.cs_category='전체'
    c1,c2=st.columns([2.3,1.3])
    with c1:
        category=st.selectbox('카테고리',category_options,key='cs_category')
    with c2:
        order=st.selectbox('정렬', ['추천순','낮은 가격순','높은 가격순'],key='cs_order')
    if st.session_state.cs_use_real_api and st.session_state.cs_search_submitted:
        try:
            data = search_api(effective_query, 'text' if mode=='텍스트 검색' else 'image')
            source = 'api'
            st.success('실제 검색 API 응답을 표시하고 있습니다.')
        except ValueError as exc:
            st.error(str(exc))
            st.caption('실제 API 모드에서는 실패 결과를 임의 데모 검색 결과로 바꿔 표시하지 않습니다.')
            return
    else:
        if source == 'demo':
            st.caption('검색 결과는 샘플 상품에 대한 키워드 필터 결과입니다. CLIP 검색 결과가 아닙니다.')
        elif source == 'csv':
            st.caption('상품 CSV에 대한 단순 키워드 필터 결과입니다. 실제 CLIP 검색과는 구분됩니다.')
    if mode in ('이미지 검색','하이브리드 검색') and st.session_state.cs_search_submitted:
        effective_query='' # do not falsely claim image similarity; browse demo catalog only
    result=filter_products(data, effective_query, category, order)
    section('검색 결과' if st.session_state.cs_search_submitted else '추천 상품 둘러보기',f'{len(result):,}개 상품 · {"UI 데모" if source=="demo" else "연결 데이터"}')
    render_grid(result,source,'search',max_items=24)


def recommend_sidebar(products: list[dict],source: str):
    st.markdown('<div class="cs-panel-title">✧ AI 추천</div>',unsafe_allow_html=True)
    st.markdown('<div class="cs-small">취향 기반 추천 UI 미리보기</div>',unsafe_allow_html=True)
    if source=='demo': st.caption('시연용 스타일 규칙 · 실제 클릭 기록 분석 아님')
    recs=suggest_products(products,'미니멀',3)
    explanations=['베이지 계열의 미니멀 코디 제안','니트와 함께 매치하기 좋은 스타일','새로운 스타일 둘러보기']
    for i,p in enumerate(recs):
        cols=st.columns([1,2.25],gap='small')
        with cols[0]:st.image(image_for(p),use_container_width=True)
        with cols[1]:
            st.markdown(f'<div class="cs-tag">{["스타일 추천", "유사 스타일", "탐색 제안"][i]}</div><div class="cs-right-card"><b>{safe(p["name"])}</b><small>{safe(explanations[i])}</small></div>',unsafe_allow_html=True)
    st.markdown('<div class="cs-outfit"><strong>오늘의 AI 코디 제안</strong><p>지금 인기 있는 스타일을 미리 살펴보세요.<br>실제 코디 생성 모델은 연결 예정입니다.</p></div>',unsafe_allow_html=True)


def metrics_teaser():
    section('서비스 현황','보유 평가 파일에 기록된 수치만 표시')
    try:
        search=load_search_reports(ROOT)
        rec=load_recommendation_metrics(ROOT)
        session=load_session_report(ROOT)
    except ValueError as exc:
        st.error(f'평가 파일 검증 오류: {exc}')
        return
    bm=search['bm25_full'] or {}
    clip=search['clip_pilot'] or {}
    lat=search['latency_pilot'] or {}
    timings=lat.get('timings') or {}
    text_p95=timings.get('text') or {}
    cols=st.columns(4,gap='small')
    cols[0].metric('BM25 MRR@10', f'{bm["metrics"]["mrr"]:.4f}' if 'mrr' in bm.get('metrics',{}) else '미제공')
    cols[1].metric('CLIP 파일럿 MRR@10',f'{clip["metrics"]["mrr"]:.4f}' if 'mrr' in clip.get('metrics',{}) else '미제공')
    cols[2].metric('추천 HitRate@50',f'{rec["hit_rate_at_50"]:.4f}' if rec and 'hit_rate_at_50' in rec else '미제공')
    cols[3].metric('텍스트 검색 p95',f'{text_p95["p95_ms"]:.1f}ms' if 'p95_ms' in text_p95 else '미제공')
    st.caption('BM25 전체 평가 · CLIP 1천 개 파일럿 · CPU 검색 함수 자체 지연으로, 서로 같은 평가 조건이 아닙니다. 추천 수치가 없으면 표시하지 않습니다.')


def home_page(products: list[dict],source: str):
    left,right=st.columns([3.5,1.13],gap='medium')
    with left:
        st.markdown('''<div class="cs-hero"><p class="cs-hero-chip">RE:SEARCH · PERSONAL STYLE</p><h1>원하는 스타일,<br>AI가 찾아드려요</h1><p>텍스트, 이미지, 그리고 둘 다 —<br>더 편리하고 똑똑한 패션 검색</p><span class="cs-pill">당신만의 스타일을 발견하세요 &nbsp; →</span></div>''',unsafe_allow_html=True)
        st.write('')
        query,mode,uploaded,submit=search_form('home')
        search_result(products,source,query,mode,uploaded,submit)
    with right:
        recommend_sidebar(products,source)
    metrics_teaser()


def search_page(products:list[dict],source:str):
    section('멀티모달 검색','텍스트 · 이미지 · 하이브리드')
    st.caption('실제 CLIP/FAISS 검색 API가 연결되기 전에는 로컬 샘플 검색만 동작합니다.')
    query,mode,uploaded,submit=search_form('search')
    search_result(products,source,query,mode,uploaded,submit)


def recommendation_page(products:list[dict],source:str):
    section('개인화 추천','Multi-Stage Recommendation')
    st.markdown('당신의 스타일에 맞춘 상품을 찾아보세요. **현재 기본 모드는 시연용 규칙 기반 추천**입니다.')
    style=st.selectbox('스타일 프로필', ['미니멀','캐주얼','페미닌'],help='실제 사용자 취향 데이터가 없으므로 시연용 프로필입니다.')
    candidate=suggest_products(products,style,8)
    active_source=source
    if st.session_state.cs_use_real_api:
        uid=st.text_input('사용자 ID',value='user_001')
        if st.button('추천 API 호출',type='primary'):
            try:
                st.session_state.cs_api_recommendations=recommend_api(uid)
                st.session_state.cs_api_rec_error=None
            except ValueError as exc:
                st.session_state.cs_api_recommendations=None
                st.session_state.cs_api_rec_error=str(exc)
        if st.session_state.get('cs_api_rec_error'): st.error(st.session_state.cs_api_rec_error)
        if st.session_state.get('cs_api_recommendations') is not None:
            candidate=st.session_state.cs_api_recommendations
            active_source='api'
    else:
        st.info(f'현재는 선택한 「{style}」 스타일에 맞춘 **샘플 상품 우선 정렬**입니다. Two-Tower/DeepFM 모델 예측 결과가 아닙니다.')
    section('당신을 위한 상품',f'{len(candidate)}개 · {"API 연동" if active_source=="api" else "시연"}')
    render_grid(candidate,active_source,'rec',max_items=12)


def main():
    page=sidebar()
    render_header(page)
    try: products,source=load_products(ROOT)
    except (OSError,UnicodeError,ValueError) as exc:
        st.error(f'상품 CSV 파일을 읽을 수 없습니다: {exc}')
        products=[];source='csv'
    if page=='홈': home_page(products,source)
    elif page=='멀티모달 검색': search_page(products,source)
    elif page=='개인화 추천': recommendation_page(products,source)
    elif page=='대시보드':
        section('운영 대시보드','검색·추천·세션 평가 결과')
        render_analytics('dashboard')
    else:
        section('실험 관리','A/B 실험 · CT 모니터링')
        render_analytics('experiments')


if __name__ == '__main__':
    main()
