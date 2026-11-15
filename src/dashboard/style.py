"""Design-system CSS for the CodiSay Streamlit showcase."""
from __future__ import annotations

import base64
from pathlib import Path


def stylesheet() -> str:
    hero = Path(__file__).resolve().parent / 'assets/hero.jpg'
    data = base64.b64encode(hero.read_bytes()).decode('ascii')
    return '''
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Noto+Sans+KR:wght@400;500;600;700;800&display=swap');
:root {--ink:#2c2827;--muted:#79716c;--coffee:#4b403b;--beige:#eae2db;--canvas:#f7f5f3;--stroke:#eae5e1;}
html,body,[data-testid="stAppViewContainer"],button,input {font-family: 'DM Sans','Noto Sans KR',sans-serif!important;}
[data-testid="stAppViewContainer"] {background:var(--canvas);color:var(--ink)}
[data-testid="stHeader"] {background:transparent;height:0px}
#MainMenu,footer {visibility:hidden}
.block-container {padding:1.45rem 1.8rem 3.8rem;max-width:1540px}
[data-testid="stSidebar"] {background:linear-gradient(180deg,#282526 0%,#211e20 100%);border-right:none}
[data-testid="stSidebar"] [data-testid="stSidebarContent"] {padding-top:1.15rem}
[data-testid="stSidebar"] p,[data-testid="stSidebar"] label,[data-testid="stSidebar"] span {color:#eee8e4!important}
[data-testid="stSidebar"] [data-testid="stRadio"] label {margin-bottom:.42rem;padding:3px 5px;border-radius:9px}
[data-testid="stSidebar"] [data-testid="stRadio"] label:has(input:checked) {background:#574d49}
[data-testid="stSidebar"] [data-testid="stRadio"] label:hover {background:#393335}
[data-testid="stSidebar"] div[data-testid="stMarkdownContainer"] a {color:#efdfd1}
[data-testid="stSidebar"] [data-testid="stVerticalBlock"] {gap:.7rem}
[data-testid="stSidebar"] .stButton button {background:#504640;border-color:#685b56;color:#fff}
[data-testid="stSidebar"] hr {border-color:#4d4544}
.cs-brand-side {font-size:25px;font-weight:800;letter-spacing:-1px;color:#fff;padding:5px 4px 20px}
.cs-brand-side small {display:block;font-size:10px;letter-spacing:4px;color:#c6bcb7;font-weight:500;margin-left:3px}
.cs-side-bottom {padding:18px 14px;border:1px solid #5a504c;border-radius:13px;background:#3a3433;color:#f1ede9;font-size:14px;font-weight:700;line-height:1.7;margin-top:15px}
.cs-side-bottom small {color:#d1c5bf;font-size:11px;font-weight:400}
.cs-header {display:flex;align-items:center;justify-content:space-between;margin-bottom:19px;padding:0 1px}
.cs-header-brand {font-size:23px;font-weight:800;letter-spacing:-1.4px;color:#262322}
.cs-header-brand span {font-size:13px;font-weight:700;letter-spacing:0;margin-left:7px}
.cs-header-sub {font-size:12px;color:#9a918b;margin-left:17px;border-left:1px solid #ddd2cb;padding-left:14px;font-weight:500}
.cs-header-side {font-size:12px;color:#7d746e;background:#fff;border:1px solid #ebe6e2;border-radius:999px;padding:8px 12px}
.cs-hero {position:relative;min-height:236px;border-radius:17px;overflow:hidden;padding:31px 36px;
background:linear-gradient(90deg,#ece2dc 0%,rgba(239,231,224,.98) 34%,rgba(239,231,224,.24) 64%,rgba(239,231,224,0) 100%),url('data:image/jpeg;base64,'''+data+''') right 46% center/auto 122% no-repeat,#ebe1da;
box-shadow:0 5px 22px rgba(42,30,20,.045)}
.cs-hero h1 {font-size:31px!important;font-weight:800;letter-spacing:-1.8px;line-height:1.35;margin:0 0 12px;color:#211d1b}
.cs-hero p {font-size:12.5px;line-height:1.85;color:#584d47;margin:0 0 18px}
.cs-hero-chip {font-size:11px!important;font-weight:700!important;letter-spacing:1.6px!important;color:#867264!important;margin-bottom:9px!important}
.cs-pill {background:#302a28;color:white;font-size:12px;border-radius:999px;padding:9px 17px;display:inline-block}
.cs-section {display:flex;align-items:center;justify-content:space-between;margin:22px 0 9px;gap:12px}
.cs-section h3 {font-size:19px;font-weight:800;letter-spacing:-.6px;margin:0;color:#24201e}
.cs-section p {color:#9a918b;font-size:11px;margin:0}
.cs-panel-title {font-size:19px;font-weight:800;letter-spacing:-.7px;color:#26211f;margin-bottom:3px}
.cs-small {font-size:11px;color:#8d8079;margin:2px 0 10px}
.cs-right-card {background:#fff;border:1px solid #ede8e3;border-radius:13px;padding:12px;margin:9px 0;box-shadow:0 6px 18px rgba(59,41,29,.035)}
.cs-tag {display:inline-block;font-size:10px;font-weight:700;color:#81655c;background:#f2e7e1;border-radius:999px;padding:4px 10px;margin-bottom:4px}
.cs-right-card b {display:block;font-size:13px;color:#2e2926;line-height:1.5}
.cs-right-card small {font-size:11px;color:#918780;line-height:1.6;display:block;margin-top:4px}
.cs-outfit {background:#eee7e0;border-radius:13px;padding:19px 15px;margin-top:19px}
.cs-outfit strong {font-size:14px;color:#302926}
.cs-outfit p {font-size:11px;line-height:1.7;color:#73655d;margin:5px 0 0}
.cs-product-title {font-size:12px;font-weight:700;line-height:1.55;color:#282522;min-height:21px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.cs-product-meta {font-weight:800;font-size:13px;color:#2b2624;margin-top:4px;white-space:nowrap}
.cs-product-detail {font-size:10px;color:#9a8982;line-height:1.8}
.cs-demo-chip {display:inline-block;background:#f4e7df;color:#8a624c;padding:3px 8px;border-radius:999px;font-size:10px;font-weight:700}
.cs-note {color:#847972;font-size:11px;margin:5px 0 9px}
.stButton>button[kind="primary"],button[data-testid="stBaseButton-primary"] {background:#352e2a;color:#fff;border-color:#352e2a}
.stButton>button {border-radius:10px;font-size:12px;font-weight:600}
[data-testid="stForm"] {border:1px solid #ede8e4;background:#fff;border-radius:16px;padding:13px 18px 17px;box-shadow:0 5px 20px rgba(54,39,26,.03)}
[data-testid="stMetric"] {background:#fff;border:1px solid #ece6e2;border-radius:13px;padding:12px 15px;min-height:100px}
[data-testid="stMetricLabel"] {font-size:11px;color:#817874}
[data-testid="stMetricValue"] {font-size:21px;color:#2c2623}
[data-testid="stVerticalBlockBorderWrapper"] > div {border-color:#ede8e4!important;border-radius:14px!important}
[data-testid="stImage"] img {border-radius:10px}
[data-testid="stAlert"] {border-radius:10px}
[data-testid="stTabs"] button p {font-size:13px}
@media(max-width:1000px){.cs-hero {background-size:auto 118%}.cs-header-sub {display:none}.block-container {padding:1.2rem .8rem 3rem}}
</style>
'''
