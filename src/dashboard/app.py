"""Unified CodiSay entrypoint: faithful storefront preview + real analytics.

The visible home screen renders the exact same HTML/CSS as codisay_preview.html,
so Streamlit native widgets can no longer distort its layout or colors.
The original Streamlit UI remains in app_native.py for backwards compatibility.
"""
from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.dashboard.storefront import storefront_html
from src.dashboard.analytics_views import render_analytics

st.set_page_config(page_title='RE:SEARCH | AI 패션 검색', page_icon='🧥',
                   layout='wide', initial_sidebar_state='collapsed')

page = st.query_params.get('page', 'home')
if isinstance(page, list):
    page=page[0] if page else 'home'

if page in ('dashboard', 'experiments'):
    st.markdown("""<style>
      :root { color-scheme: light; }
      [data-testid="stAppViewContainer"] {background:#f8f6f4;color:#292624}
      .block-container {max-width:1400px;padding-top:2rem}
      [data-testid="stHeader"] {background:#f8f6f4}
    </style>""",unsafe_allow_html=True)
    if st.button('← RE:SEARCH 쇼핑 화면으로'):
        st.query_params.clear()
        st.rerun()
    st.title('RE:SEARCH · ' + ('운영 대시보드' if page=='dashboard' else '실험관리'))
    st.caption('이 화면은 기존 Streamlit 분석 기능을 그대로 사용합니다. 홈 화면은 HTML 미리보기와 동일한 디자인입니다.')
    render_analytics('dashboard' if page=='dashboard' else 'experiments')
else:
    # No Streamlit sidebar or widget layout: pixel-identical frontend iframe.
    st.markdown("""<style>
      :root {color-scheme:light;}
      [data-testid="stAppViewContainer"] {background:#f8f6f4}
      [data-testid="stHeader"], [data-testid="stSidebar"],
      [data-testid="stToolbar"], #MainMenu, footer {display:none!important}
      [data-testid="stAppViewBlockContainer"], .block-container {
        max-width:100%!important;width:100%!important;padding:0!important;margin:0!important;
      }
      [data-testid="stVerticalBlock"], [data-testid="stElementContainer"] {
        gap:0!important;margin:0!important;padding:0!important;
      }
      iframe {border:0!important;display:block!important}
      [data-testid="stCustomComponentV1"] {width:100%!important}
    </style>""",unsafe_allow_html=True)
    try:
        markup, source=storefront_html(ROOT,embedded=True)
    except (OSError,UnicodeError,ValueError) as exc:
        st.error(f'쇼핑 화면 데이터 로딩 오류: {exc}')
    else:
        components.html(markup, height=1000, scrolling=True)
