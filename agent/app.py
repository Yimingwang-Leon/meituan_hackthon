from __future__ import annotations

import os
import sys
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent))
load_dotenv(Path(__file__).parent / ".env")

from src.agent import OUTBOUND_INSTRUCTIONS, make_outbound_agent
from src.evaluator import (
    EvaluationReport,
    compute_coverage,
    evaluate_session,
)
from src.memory import append_evaluation_memory
from src.orders import load_pending_orders
from src.persona import ALL_PERSONAS
from src.rule_parser import parse_rules
from src.session import OutboundSession
from src.simulator import UserSimulator

MAX_TURNS = 15
SESSIONS_DIR = Path(__file__).parent / "sessions"
ORDERS_DIR = Path(__file__).parent / "orders"
MEMORY_DIR = Path(__file__).parent / "memory"

PERSONA_LABELS = {
    "cooperative": "配合型",
    "suspicious": "警惕型",
    "impatient": "急躁型",
    "ambiguous": "模糊型",
    "info_missing": "缺信息型",
    "rejector": "拒收型",
}

st.set_page_config(
    page_title="外呼数字人评测系统",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ═══════════════════════════════════════════════════════════════════════════
# 设计系统 · 浅色 / 蓝调 / 高对比
# ═══════════════════════════════════════════════════════════════════════════
CUSTOM_CSS = """
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&family=Noto+Serif+SC:wght@600;700&display=swap" rel="stylesheet">

<style>
/* ── 隐藏 Streamlit 默认 chrome ─────────────────────────────────── */
#MainMenu, footer { display: none !important; }
header[data-testid="stHeader"] { background: transparent; height: 0; }
[data-testid="stToolbar"] { display: none; }

/* ── Design tokens ─────────────────────────────────────────────── */
:root {
    --bg:           #fafaf7;
    --surface:      #ffffff;
    --surface-hi:   #fcfbf5;
    --border:       #e7e5e0;
    --border-2:     #d6d3cc;

    --text:         #1c1917;
    --text-2:       #44403c;
    --text-3:       #78716c;
    --text-4:       #a8a29e;

    --brand:        #FFC300;   /* 美团黄 */
    --brand-2:      #b45309;   /* 深琥珀，用于文字/边框 */
    --brand-3:      #92400e;   /* 更深，用于强对比 */
    --brand-soft:   #fffbeb;
    --brand-mid:    #fef3c7;
    --brand-glow:   rgba(255, 195, 0, 0.30);

    --accent:       #f59e0b;
    --accent-soft:  #fef3c7;

    --success:      #059669;
    --success-soft: #ecfdf5;
    --warning:      #ea580c;
    --warning-soft: #fff7ed;
    --danger:       #dc2626;
    --danger-soft:  #fef2f2;

    --serif: "Noto Serif SC", "Songti SC", "STSong", serif;
    --sans:  "Inter", -apple-system, BlinkMacSystemFont, "Segoe UI",
             "PingFang SC", "Microsoft YaHei", "Helvetica Neue", sans-serif;
    --mono:  "JetBrains Mono", "SF Mono", Menlo, Consolas, monospace;
}

/* ── 全局 ──────────────────────────────────────────────────────── */
html, body, [class*="css"], .stApp {
    background: var(--bg) !important;
    color: var(--text) !important;
    font-family: var(--sans) !important;
    -webkit-font-smoothing: antialiased;
    -moz-osx-font-smoothing: grayscale;
}
.stApp {
    background:
        radial-gradient(ellipse 90% 50% at 50% -20%, var(--brand-glow), transparent 70%),
        var(--bg) !important;
}
.block-container {
    padding-top: 2.5rem !important;
    padding-bottom: 6rem !important;
    max-width: 1440px;
}

/* ── 字体层级 ──────────────────────────────────────────────────── */
h1, h2, h3, h4, h5, h6 { color: var(--text) !important; }
h1 { font-size: 2rem !important;   font-weight: 700 !important; letter-spacing: -0.03em !important; }
h2 { font-size: 1.375rem !important; font-weight: 650 !important; letter-spacing: -0.015em !important; }
h3 { font-size: 1rem !important;   font-weight: 600 !important; letter-spacing: -0.01em !important; }

p, span, div, label { color: var(--text); }
.stCaption, [data-testid="stCaptionContainer"] {
    color: var(--text-3) !important;
    font-size: 0.8125rem !important;
}

/* ── 按钮 ──────────────────────────────────────────────────────── */
.stButton > button {
    background: var(--surface) !important;
    color: var(--text) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    font-family: var(--sans) !important;
    font-weight: 500 !important;
    font-size: 0.9rem !important;
    transition: all 0.15s ease !important;
}
.stButton > button:hover {
    border-color: var(--brand) !important;
    color: var(--brand) !important;
}
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #FFD60A 0%, #FFC300 50%, #FFA500 100%) !important;
    color: #1c1917 !important;
    border: 1px solid #d97706 !important;
    font-weight: 700 !important;
    font-size: 0.95rem !important;
    height: 52px !important;
    letter-spacing: 0.01em !important;
    box-shadow:
        0 1px 2px rgba(28,25,23,0.06),
        0 10px 24px -8px var(--brand-glow) !important;
    transition: all 0.2s cubic-bezier(0.16, 1, 0.3, 1) !important;
}
.stButton > button[kind="primary"]:hover:not(:disabled) {
    transform: translateY(-1px);
    box-shadow:
        0 4px 10px rgba(15,23,42,0.06),
        0 16px 36px -10px var(--brand-glow) !important;
}
.stButton > button[kind="primary"]:disabled {
    background: var(--bg) !important;
    color: var(--text-4) !important;
    border-color: var(--border) !important;
    box-shadow: none !important;
}

/* ── 输入控件 ──────────────────────────────────────────────────── */
[data-testid="stTextArea"] textarea,
[data-testid="stTextInput"] input {
    background: var(--surface) !important;
    color: var(--text) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
    font-family: var(--mono) !important;
    font-size: 0.8125rem !important;
    line-height: 1.7 !important;
    padding: 1rem !important;
    transition: all 0.15s ease !important;
}
[data-testid="stTextArea"] textarea:focus,
[data-testid="stTextInput"] input:focus {
    border-color: var(--brand) !important;
    outline: none !important;
    box-shadow: 0 0 0 3px var(--brand-glow) !important;
}

.stMultiSelect [data-baseweb="select"] > div,
.stSelectbox [data-baseweb="select"] > div {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
}
.stMultiSelect [data-baseweb="tag"] {
    background: var(--brand-mid) !important;
    color: var(--brand-3) !important;
    border: 1px solid #fcd34d !important;
    border-radius: 6px !important;
    font-size: 0.8125rem !important;
    font-weight: 500 !important;
}
.stMultiSelect [data-baseweb="tag"] svg { fill: var(--brand-3) !important; }

/* ── 复选框 ────────────────────────────────────────────────────── */
.stCheckbox > label {
    color: var(--text) !important;
    font-size: 0.9375rem !important;
}
.stCheckbox [data-baseweb="checkbox"] > div:first-child {
    background-color: var(--surface) !important;
    border-color: var(--border-2) !important;
    border-radius: 4px !important;
}
.stCheckbox [data-baseweb="checkbox"][aria-checked="true"] > div:first-child {
    background-color: var(--brand) !important;
    border-color: var(--brand) !important;
}

/* ── Metric ────────────────────────────────────────────────────── */
[data-testid="stMetric"] {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1rem 1.25rem;
}
[data-testid="stMetricValue"] {
    color: var(--text) !important;
    font-size: 1.875rem !important;
    font-weight: 700 !important;
    letter-spacing: -0.025em !important;
    font-variant-numeric: tabular-nums !important;
}
[data-testid="stMetricLabel"] {
    color: var(--text-3) !important;
    font-size: 0.8125rem !important;
}

/* ── Expander ──────────────────────────────────────────────────── */
details[data-testid="stExpander"] {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
    margin-bottom: 0.5rem !important;
    overflow: hidden;
    box-shadow: 0 1px 2px rgba(15,23,42,0.02);
}
details[data-testid="stExpander"][open] {
    border-color: var(--border-2) !important;
    box-shadow: 0 4px 14px rgba(15,23,42,0.04);
}
details[data-testid="stExpander"] summary {
    padding: 0.95rem 1.125rem !important;
    background: var(--surface) !important;
    color: var(--text) !important;
    font-size: 0.9375rem !important;
    font-weight: 500 !important;
    transition: background 0.15s !important;
}
details[data-testid="stExpander"] summary:hover {
    background: var(--surface-hi) !important;
}

/* ── Progress bar ──────────────────────────────────────────────── */
[data-testid="stProgress"] > div > div {
    background: var(--border) !important;
    border-radius: 999px !important;
    height: 5px !important;
}
[data-testid="stProgress"] > div > div > div {
    background: linear-gradient(90deg, #FFD60A, #FFA500) !important;
    border-radius: 999px !important;
}
[data-testid="stProgress"] p {
    color: var(--text-3) !important;
    font-family: var(--mono) !important;
    font-size: 0.8125rem !important;
}

/* ── 对话气泡 ──────────────────────────────────────────────────── */
[data-testid="stChatMessage"] {
    background: var(--surface-hi) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
    color: var(--text) !important;
    padding: 0.625rem 0.875rem !important;
    margin-bottom: 0.5rem !important;
    box-shadow: none !important;
}
[data-testid="stChatMessage"] p {
    color: var(--text) !important;
    font-size: 0.9rem !important;
    line-height: 1.5 !important;
}

/* ── 提示框 ────────────────────────────────────────────────────── */
[data-testid="stAlertContainer"] {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
    color: var(--text) !important;
}

/* ── 侧边栏 ────────────────────────────────────────────────────── */
[data-testid="stSidebar"] {
    background: var(--surface) !important;
    border-right: 1px solid var(--border);
}
[data-testid="stSidebar"] .block-container { padding: 2rem 1.25rem !important; }
[data-testid="stSidebar"] h3 {
    font-size: 0.8125rem !important;
    color: var(--text-3) !important;
    margin: 1.5rem 0 0.625rem 0 !important;
    font-weight: 600 !important;
    letter-spacing: 0.02em !important;
}

/* ── 分隔线 ────────────────────────────────────────────────────── */
hr {
    border: none !important;
    border-top: 1px solid var(--border) !important;
    margin: 2.5rem 0 !important;
}

/* ═══════════════════════════════════════════════════════════════════ */
/* 自定义组件                                                          */
/* ═══════════════════════════════════════════════════════════════════ */

.brand-mark {
    display: inline-flex;
    align-items: center;
    gap: 0.5rem;
    padding: 0.4rem 0.875rem;
    border: 1px solid var(--border);
    background: var(--surface);
    border-radius: 999px;
    font-size: 0.8125rem;
    color: var(--text-2);
    font-weight: 500;
    margin-bottom: 1.5rem;
    box-shadow: 0 1px 2px rgba(15,23,42,0.03);
}
.brand-mark .dot {
    width: 7px; height: 7px;
    background: #FFB300;
    border-radius: 50%;
    box-shadow: 0 0 0 3px var(--brand-glow);
    animation: pulse 2s ease-in-out infinite;
}
@keyframes pulse {
    0%, 100% { transform: scale(1); }
    50%      { transform: scale(0.85); }
}

/* — Hero — */
.hero {
    padding: 0.5rem 0 3rem 0;
    border-bottom: 1px solid var(--border);
    margin-bottom: 2.5rem;
}
.hero-headline {
    font-size: clamp(2.25rem, 4.2vw, 3.75rem);
    line-height: 1.1;
    letter-spacing: -0.035em;
    font-weight: 700;
    margin: 0 0 1.25rem 0;
    color: var(--text);
}
.hero-headline em {
    font-style: italic;
    font-family: var(--serif);
    font-weight: 700;
    letter-spacing: -0.02em;
    background: linear-gradient(135deg, #d97706 0%, #ea580c 50%, #b45309 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}
.hero-sub {
    font-size: 1.0625rem;
    color: var(--text-2);
    max-width: 720px;
    line-height: 1.65;
    margin: 0;
}
.hero-meta {
    display: flex;
    gap: 2.5rem;
    margin-top: 2rem;
    padding-top: 1.75rem;
    border-top: 1px solid var(--border);
}
.hero-meta-item .label {
    font-size: 0.75rem;
    color: var(--text-3);
    margin-bottom: 0.375rem;
    font-weight: 500;
}
.hero-meta-item .value {
    font-size: 1rem;
    color: var(--text);
    font-weight: 600;
    letter-spacing: -0.01em;
}

/* — 编号章节 — */
.section {
    display: grid;
    grid-template-columns: 72px 1fr auto;
    align-items: baseline;
    gap: 1.5rem;
    margin: 3.5rem 0 1.25rem 0;
    padding-top: 1.75rem;
    border-top: 1px solid var(--border);
}
.section-num {
    font-family: var(--mono);
    font-size: 0.875rem;
    color: var(--brand-2);
    font-weight: 600;
    letter-spacing: 0.04em;
}
.section-title {
    font-size: 1.625rem;
    font-weight: 700;
    letter-spacing: -0.025em;
    color: var(--text);
    margin: 0;
}
.section-title em {
    font-family: var(--serif);
    font-style: italic;
    font-weight: 700;
    color: var(--brand-2);
}
.section-meta {
    font-size: 0.8125rem;
    color: var(--text-3);
    text-align: right;
    font-weight: 500;
}

/* — Bento — */
.bento {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 1rem;
    margin: 1rem 0;
}
.bento-cell {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 1.625rem 1.5rem;
    min-height: 156px;
    box-shadow: 0 1px 2px rgba(15,23,42,0.02);
    transition: all 0.2s;
}
.bento-cell:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 24px -8px rgba(15,23,42,0.08);
    border-color: var(--border-2);
}
.bento-cell.feature {
    grid-column: span 2;
    background:
        radial-gradient(circle at 100% 0%, var(--brand-glow), transparent 55%),
        linear-gradient(135deg, #ffffff 0%, var(--brand-soft) 100%);
    border-color: #fcd34d;
}
.bento-label {
    font-size: 0.8125rem;
    color: var(--text-3);
    font-weight: 600;
    margin-bottom: 0.875rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}
.bento-label::before {
    content: "";
    width: 6px; height: 6px;
    background: var(--text-4);
    border-radius: 50%;
}
.bento-cell.feature .bento-label::before {
    background: var(--brand);
    box-shadow: 0 0 0 3px var(--brand-glow);
}
.bento-value {
    font-size: 3rem;
    line-height: 1;
    font-weight: 700;
    letter-spacing: -0.035em;
    color: var(--text);
    font-variant-numeric: tabular-nums;
    margin-bottom: 0.5rem;
}
.bento-cell.feature .bento-value {
    font-size: 4rem;
    background: linear-gradient(135deg, #b45309 0%, #d97706 50%, #ea580c 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}
.bento-hint {
    font-size: 0.8125rem;
    color: var(--text-3);
    font-weight: 500;
}

/* — 胶囊徽章 — */
.pill {
    display: inline-flex;
    align-items: center;
    gap: 0.25rem;
    padding: 0.25rem 0.625rem;
    border-radius: 999px;
    font-size: 0.75rem;
    font-weight: 500;
    line-height: 1.4;
    border: 1px solid transparent;
    white-space: nowrap;
}
.pill-success { background: var(--success-soft); color: var(--success); border-color: #a7f3d0; }
.pill-warning { background: var(--warning-soft); color: var(--warning); border-color: #fde68a; }
.pill-danger  { background: var(--danger-soft);  color: var(--danger);  border-color: #fecaca; }
.pill-neutral { background: var(--surface-hi);   color: var(--text-2);  border-color: var(--border); }
.pill-brand   { background: var(--brand-soft);   color: var(--brand-3); border-color: #fcd34d; }

/* — 规则列表 — */
.rule-list { display: flex; flex-direction: column; }
.rule-item {
    display: grid;
    grid-template-columns: 24px 1fr;
    gap: 0.75rem;
    padding: 0.75rem 0;
    border-bottom: 1px solid var(--border);
    align-items: flex-start;
}
.rule-item:last-child { border-bottom: none; }
.rule-mark {
    width: 20px; height: 20px;
    border-radius: 5px;
    display: flex; align-items: center; justify-content: center;
    font-size: 0.8rem;
    font-weight: 700;
    margin-top: 2px;
}
.rule-mark.pass { background: var(--success-soft); color: var(--success); border: 1px solid #a7f3d0; }
.rule-mark.fail { background: var(--danger-soft);  color: var(--danger);  border: 1px solid #fecaca; }
.rule-mark.na   { background: var(--surface-hi);   color: var(--text-4);  border: 1px solid var(--border); }
.rule-text {
    font-size: 0.9rem;
    color: var(--text);
    line-height: 1.5;
    font-weight: 500;
}
.rule-meta-row {
    display: flex;
    gap: 0.375rem;
    align-items: center;
    margin-top: 0.5rem;
    flex-wrap: wrap;
}
.rule-meta-row .id {
    font-family: var(--mono);
    font-size: 0.7rem;
    color: var(--text-3);
    padding: 0.15rem 0.4rem;
    background: var(--surface-hi);
    border-radius: 4px;
    border: 1px solid var(--border);
}
.rule-evidence {
    font-size: 0.85rem;
    color: var(--text-2);
    margin-top: 0.5rem;
    padding: 0.5rem 0.75rem;
    border-left: 3px solid #f59e0b;
    background: var(--brand-soft);
    border-radius: 0 6px 6px 0;
    line-height: 1.55;
}

/* — 侧边栏小卡片 — */
.side-stat {
    background: linear-gradient(135deg, var(--brand-soft) 0%, #ffffff 100%);
    border: 1px solid #fcd34d;
    border-radius: 12px;
    padding: 1.125rem 1.25rem;
    margin-top: 0.5rem;
}
.side-stat .label {
    font-size: 0.75rem;
    color: var(--brand-3);
    font-weight: 600;
}
.side-stat .value {
    font-size: 2rem;
    font-weight: 700;
    color: var(--brand-3);
    letter-spacing: -0.03em;
    font-variant-numeric: tabular-nums;
    line-height: 1;
    margin-top: 0.4rem;
}
.side-stat .hint {
    font-size: 0.75rem;
    color: var(--text-3);
    margin-top: 0.375rem;
}

/* — 标签云 — */
.tag-cloud { display: flex; flex-wrap: wrap; gap: 0.375rem; }

/* — 会话标题中的内联标签 — */
.inline-tag {
    font-family: var(--mono);
    font-size: 0.75rem;
    color: var(--text-3);
}

/* — 段落标签 — */
.subhead {
    font-size: 0.8125rem;
    color: var(--text-3);
    font-weight: 600;
    margin-bottom: 0.75rem;
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════
# Altair 主题 · 浅色蓝调
# ═══════════════════════════════════════════════════════════════════════════
@alt.theme.register("eval_light", enable=True)
def _eval_light_theme():
    return alt.theme.ThemeConfig(
        config={
            "background": "#ffffff",
            "view": {"strokeWidth": 0, "fill": "#ffffff"},
            "axis": {
                "labelColor": "#44403c",
                "titleColor": "#1c1917",
                "labelFont": "Inter, sans-serif",
                "titleFont": "Inter, sans-serif",
                "labelFontSize": 11,
                "titleFontSize": 12,
                "labelFontWeight": 400,
                "titleFontWeight": 600,
                "domainColor": "#e7e5e0",
                "tickColor": "#e7e5e0",
                "grid": True,
                "gridColor": "#f5f5f1",
                "gridOpacity": 1,
            },
            "legend": {
                "labelColor": "#44403c",
                "titleColor": "#1c1917",
                "labelFont": "Inter, sans-serif",
                "titleFont": "Inter, sans-serif",
                "labelFontSize": 11,
                "titleFontSize": 11,
                "titleFontWeight": 600,
            },
            "title": {"color": "#1c1917", "fontSize": 13, "fontWeight": 600},
            "range": {"category": ["#FFB300", "#d97706", "#059669", "#dc2626", "#7c3aed", "#0891b2"]},
        }
    )


# ═══════════════════════════════════════════════════════════════════════════
# 状态
# ═══════════════════════════════════════════════════════════════════════════
for key, default in [
    ("reports", []),
    ("transcripts", []),
    ("parsed_rules", []),
    ("run_complete", False),
]:
    if key not in st.session_state:
        st.session_state[key] = default


# ═══════════════════════════════════════════════════════════════════════════
# 渲染辅助
# ═══════════════════════════════════════════════════════════════════════════
def _score_pill(score: float) -> str:
    if score >= 0.8: return f'<span class="pill pill-success">通过 · {score:.0%}</span>'
    if score >= 0.5: return f'<span class="pill pill-warning">待改进 · {score:.0%}</span>'
    return f'<span class="pill pill-danger">薄弱 · {score:.0%}</span>'


def _conf_pill(c: float) -> str:
    if c >= 0.99: return f'<span class="pill pill-success">置信度 {c:.0%}</span>'
    if c >= 0.66: return f'<span class="pill pill-warning">置信度 {c:.0%}</span>'
    return f'<span class="pill pill-danger">置信度 {c:.0%}</span>'


def _sev_pill(sev: str) -> str:
    label = {"critical": "致命", "major": "严重", "minor": "轻微"}.get(sev, sev)
    cls = {"critical": "pill-danger", "major": "pill-warning", "minor": "pill-neutral"}.get(sev, "pill-neutral")
    return f'<span class="pill {cls}">{label}</span>'


def _persona_label(persona_type: str) -> str:
    return PERSONA_LABELS.get(persona_type, persona_type)


def _section_head(num: str, title_html: str, meta: str = "") -> None:
    st.markdown(
        f'<div class="section">'
        f'  <div class="section-num">— {num}</div>'
        f'  <h2 class="section-title">{title_html}</h2>'
        f'  <div class="section-meta">{meta}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )


def _bento_cell(label: str, value: str, hint: str = "", feature: bool = False) -> str:
    cls = "bento-cell feature" if feature else "bento-cell"
    hint_html = f'<div class="bento-hint">{hint}</div>' if hint else ""
    return (
        f'<div class="{cls}">'
        f'  <div class="bento-label">{label}</div>'
        f'  <div class="bento-value">{value}</div>'
        f'  {hint_html}'
        f'</div>'
    )


def _build_heatmap_df(reports: list[EvaluationReport]) -> pd.DataFrame:
    rows = []
    for report in reports:
        for rr in report.rule_results:
            score = 1.0 if rr.result == "pass" else 0.0 if rr.result == "fail" else None
            rows.append({
                "规则": f"{rr.rule_id}  {rr.description[:26]}",
                "rule_id": rr.rule_id,
                "severity": rr.severity,
                "角色": _persona_label(report.persona_type),
                "score": score,
            })
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    return (
        df.dropna(subset=["score"])
        .groupby(["规则", "rule_id", "severity", "角色"])
        .agg(通过率=("score", "mean"), 样本数=("score", "count"))
        .reset_index()
    )


def _render_heatmap(reports: list[EvaluationReport]) -> None:
    df = _build_heatmap_df(reports)
    if df.empty:
        st.caption("无可视化数据")
        return

    height = max(280, 30 * df["规则"].nunique())
    chart = (
        alt.Chart(df)
        .mark_rect(stroke="#ffffff", strokeWidth=2, cornerRadius=4)
        .encode(
            x=alt.X("角色:N", title=None, axis=alt.Axis(
                labelAngle=0, labelPadding=10, ticks=False, domain=False, orient="top",
            )),
            y=alt.Y("规则:N", title=None, sort=alt.SortField("rule_id"),
                    axis=alt.Axis(labelLimit=340, labelPadding=10, ticks=False, domain=False)),
            color=alt.Color(
                "通过率:Q",
                title="通过率",
                scale=alt.Scale(range=["#dc2626", "#f59e0b", "#059669"], domain=[0, 0.5, 1]),
                legend=alt.Legend(orient="bottom", direction="horizontal", gradientLength=220, titlePadding=10),
            ),
            tooltip=[
                alt.Tooltip("规则:N"),
                alt.Tooltip("角色:N"),
                alt.Tooltip("severity:N", title="严重度"),
                alt.Tooltip("通过率:Q", format=".0%"),
                alt.Tooltip("样本数:Q"),
            ],
        )
        .properties(height=height)
    )
    st.altair_chart(chart, width="stretch")


def _render_persona_bar(reports: list[EvaluationReport]) -> None:
    by_persona: dict[str, list[float]] = {}
    for r in reports:
        by_persona.setdefault(r.persona_type, []).append(r.score)
    df = pd.DataFrame([
        {"角色": _persona_label(k), "得分": sum(v) / len(v)}
        for k, v in by_persona.items()
    ])

    chart = (
        alt.Chart(df)
        .mark_bar(cornerRadiusEnd=4, height=22)
        .encode(
            y=alt.Y("角色:N", sort="-x", title=None,
                    axis=alt.Axis(labelPadding=10, ticks=False, domain=False)),
            x=alt.X("得分:Q", scale=alt.Scale(domain=[0, 1]),
                    axis=alt.Axis(format="%", title=None, grid=True)),
            color=alt.Color(
                "得分:Q",
                scale=alt.Scale(range=["#dc2626", "#f59e0b", "#059669"], domain=[0, 0.5, 1]),
                legend=None,
            ),
            tooltip=[alt.Tooltip("角色:N"), alt.Tooltip("得分:Q", format=".0%")],
        )
        .properties(height=max(200, 42 * len(df)))
    )
    st.altair_chart(chart, width="stretch")


def _render_session_card(item: dict, report: EvaluationReport) -> None:
    def _badge(score: float) -> str:
        if score >= 0.8: return "通过"
        if score >= 0.5: return "待改进"
        return "薄弱"

    persona_label = _persona_label(report.persona_type)
    head = (
        f"{report.order_id} · {persona_label}"
        f"   ·   {_badge(report.score)} {report.score:.0%}"
        f"   ·   置信度 {report.mean_confidence:.0%}"
    )

    with st.expander(head, expanded=False):
        col_l, col_r = st.columns([1, 1.15])

        with col_l:
            st.markdown('<div class="subhead">对话记录</div>', unsafe_allow_html=True)
            for entry in item["transcript"]:
                role = "assistant" if entry.speaker == "agent" else "user"
                with st.chat_message(role):
                    st.write(entry.text)

        with col_r:
            st.markdown('<div class="subhead">规则评测</div>', unsafe_allow_html=True)
            rows = []
            for rr in report.rule_results:
                mark_cls = {"pass": "pass", "fail": "fail", "not_applicable": "na"}[rr.result]
                mark_char = {"pass": "✓", "fail": "✗", "not_applicable": "—"}[rr.result]
                votes = " ".join(
                    f'<span class="pill pill-neutral">{k}×{v}</span>'
                    for k, v in rr.votes.items()
                )
                evidence = (
                    f'<div class="rule-evidence">{rr.evidence}</div>'
                    if rr.result != "not_applicable" else ""
                )
                rows.append(
                    f'<div class="rule-item">'
                    f'  <div class="rule-mark {mark_cls}">{mark_char}</div>'
                    f'  <div>'
                    f'    <div class="rule-text">{rr.description}</div>'
                    f'    <div class="rule-meta-row">'
                    f'      <span class="id">{rr.rule_id}</span>'
                    f'      {_sev_pill(rr.severity)}'
                    f'      {_conf_pill(rr.confidence)}'
                    f'      {votes}'
                    f'    </div>'
                    f'    {evidence}'
                    f'  </div>'
                    f'</div>'
                )
            st.markdown(f'<div class="rule-list">{"".join(rows)}</div>', unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════
# 侧边栏
# ═══════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown(
        '<div class="brand-mark"><span class="dot"></span>评测控制台</div>',
        unsafe_allow_html=True,
    )

    all_orders = load_pending_orders(ORDERS_DIR)
    order_options = [o.order.order_id for o in all_orders]

    st.markdown("### 订单")
    selected_order_ids = st.multiselect(
        " ",
        order_options,
        default=order_options[:1] if order_options else [],
        label_visibility="collapsed",
        placeholder="请选择待评测订单...",
    )

    st.markdown("### 用户角色")
    selected_personas = []
    for persona in ALL_PERSONAS:
        checked = st.checkbox(
            _persona_label(persona.persona_type),
            value=True,
            key=f"p_{persona.persona_type}",
        )
        if checked:
            selected_personas.append(persona)

    st.markdown("### 评测深度")
    judge_samples = st.select_slider(
        " ",
        options=[1, 2, 3],
        value=1,
        format_func=lambda x: f"{x} 次采样" + (" · 快" if x == 1 else " · 稳" if x == 3 else ""),
        label_visibility="collapsed",
        help="每条规则的 LLM 评判次数。多采样可以输出置信度，但慢约 N 倍。",
    )

    n_sessions = len(selected_order_ids) * len(selected_personas)
    st.markdown(
        f'<div class="side-stat">'
        f'  <div class="label">即将运行</div>'
        f'  <div class="value">{n_sessions} 个会话</div>'
        f'  <div class="hint">{len(selected_order_ids)} 订单 × {len(selected_personas)} 角色</div>'
        f'</div>',
        unsafe_allow_html=True,
    )


# ═══════════════════════════════════════════════════════════════════════════
# Hero
# ═══════════════════════════════════════════════════════════════════════════
st.markdown(
    '<div class="brand-mark"><span class="dot"></span>美团 · 履约外呼数字人评测系统</div>',
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero">
        <h1 class="hero-headline">
            自动评估对话模型的<br/>
            <em>指令遵循能力</em>。
        </h1>
        <p class="hero-sub">
            粘贴任意外呼任务指令，系统自动拆解为原子规则，调用多种用户角色模拟真实对话，
            通过多采样 LLM 评判，输出可解释、可量化的评测报告。
        </p>
        <div class="hero-meta">
            <div class="hero-meta-item">
                <div class="label">评估方法</div>
                <div class="value">原子规则分解</div>
            </div>
            <div class="hero-meta-item">
                <div class="label">可靠性</div>
                <div class="value">多采样投票</div>
            </div>
            <div class="hero-meta-item">
                <div class="label">覆盖度</div>
                <div class="value">6 种角色模拟</div>
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ═══════════════════════════════════════════════════════════════════════════
# 01 · 任务指令
# ═══════════════════════════════════════════════════════════════════════════
_section_head("01", "任务<em>指令</em>", "粘贴任意外呼数字人 prompt")

instructions = st.text_area(
    " ",
    value=OUTBOUND_INSTRUCTIONS,
    height=280,
    label_visibility="collapsed",
)

col_btn, _ = st.columns([1, 3])
with col_btn:
    run_clicked = st.button(
        "▸  开始评测",
        type="primary",
        disabled=not selected_personas or not selected_order_ids,
        width="stretch",
    )


# ═══════════════════════════════════════════════════════════════════════════
# 运行流程
# ═══════════════════════════════════════════════════════════════════════════
if run_clicked:
    if not os.getenv("OPENAI_API_KEY"):
        st.error("缺少 OPENAI_API_KEY，请在 agent/.env 中填写")
        st.stop()

    orders = [o for o in all_orders if o.order.order_id in selected_order_ids]

    with st.spinner("正在解析任务指令为原子规则..."):
        parsed_rules = parse_rules(instructions)
    st.session_state.parsed_rules = parsed_rules

    n_required = sum(1 for r in parsed_rules if r.rule_type == "required")
    n_cond = sum(1 for r in parsed_rules if r.rule_type == "conditional")
    n_forbid = sum(1 for r in parsed_rules if r.rule_type == "forbidden")

    _section_head("02", "解析<em>规则</em>", "rule_parser 输出")
    st.markdown(
        f'<div class="bento">'
        f'{_bento_cell("规则总数", str(len(parsed_rules)), "原子可独立验证", feature=True)}'
        f'{_bento_cell("必做规则", str(n_required), "每次都要遵守")}'
        f'{_bento_cell("条件规则", str(n_cond), "特定场景触发")}'
        f'{_bento_cell("禁止规则", str(n_forbid), "任何时候禁止")}'
        f'</div>',
        unsafe_allow_html=True,
    )

    agent = make_outbound_agent(instructions)
    st.session_state.reports = []
    st.session_state.transcripts = []
    st.session_state.run_complete = False

    tasks = [(order, persona) for order in orders for persona in selected_personas]
    total = len(tasks)

    _section_head("03", "实时<em>执行</em>", f"{total} 个会话 · 顺序运行")

    progress = st.progress(0, text=f"准备运行 {total} 个会话...")
    live = st.container()

    for done, (order, persona) in enumerate(tasks):
        persona_label = _persona_label(persona.persona_type)
        label = f"{order.order.order_id} × {persona_label}"
        print(f"\n[{done + 1}/{total}] 开始 {label}", flush=True)
        progress.progress(
            done / total,
            text=f"对话生成中 · {label}  ({done + 1}/{total})",
        )

        outbound = OutboundSession(order, agent=agent)
        simulator = UserSimulator(order, persona)
        print(f"  agent.start()", flush=True)
        agent_turn = outbound.start()
        for turn_idx in range(MAX_TURNS):
            if agent_turn.should_end:
                print(f"  agent 结束于第 {turn_idx + 1} 轮", flush=True)
                break
            user_turn = simulator.reply(agent_turn.reply_text)
            if user_turn.should_end:
                outbound.record_user(user_turn.reply_text)
                print(f"  user 结束于第 {turn_idx + 1} 轮", flush=True)
                break
            agent_turn = outbound.reply(user_turn.reply_text)
        else:
            print(f"  达到 MAX_TURNS={MAX_TURNS}", flush=True)

        print(f"  开始评测 {len(parsed_rules)} 条规则 × {judge_samples} 采样", flush=True)
        progress.progress(
            done / total,
            text=f"规则评测中 · {label}  ({done + 1}/{total})",
        )
        outbound.save_archive(SESSIONS_DIR, "agent_end", persona_type=persona.persona_type)
        archive = outbound.get_archive(persona_type=persona.persona_type)
        report = evaluate_session(
            archive,
            persona.persona_type,
            rules=parsed_rules,
            n_samples=judge_samples,
            max_workers=8,
        )
        append_evaluation_memory(archive, report, MEMORY_DIR)
        print(f"  评测完成，得分 {report.score:.0%}", flush=True)
        item = {
            "order_id": order.order.order_id,
            "persona_type": persona.persona_type,
            "transcript": archive.transcript,
        }

        st.session_state.reports.append(report)
        st.session_state.transcripts.append(item)
        with live:
            _render_session_card(item, report)

        progress.progress((done + 1) / total, text=f"已完成 · {label}  ({done + 1}/{total})")

    progress.progress(1.0, text="评测完成 · 100%")
    st.session_state.run_complete = True


# ═══════════════════════════════════════════════════════════════════════════
# 仪表板
# ═══════════════════════════════════════════════════════════════════════════
if st.session_state.run_complete and st.session_state.reports:
    reports: list[EvaluationReport] = st.session_state.reports
    parsed_rules = st.session_state.parsed_rules
    total = len(reports)

    coverage = compute_coverage(reports, parsed_rules)
    avg_score = sum(r.score for r in reports) / total
    avg_conf = sum(r.mean_confidence for r in reports) / total

    _section_head("04", "评测<em>仪表板</em>", "全局指标汇总")
    st.markdown(
        f'<div class="bento">'
        f'{_bento_cell("综合得分", f"{avg_score:.0%}", f"共 {total} 个会话平均", feature=True)}'
        f'{_bento_cell("条件覆盖率", f"{coverage.coverage_rate:.0%}", f"{coverage.triggered_conditional}/{coverage.total_conditional} 条规则被触发")}'
        f'{_bento_cell("评估置信度", f"{avg_conf:.0%}", "LLM 自一致率")}'
        f'{_bento_cell("会话总数", f"{total}", "订单 × 角色")}'
        f'</div>',
        unsafe_allow_html=True,
    )

    _section_head("05", "规则 × <em>角色</em>热力图", "红=低通过率 / 绿=高通过率")
    _render_heatmap(reports)

    _section_head("06", "<em>角色</em>得分排名", "哪类用户最容易暴露问题")
    _render_persona_bar(reports)

    _section_head("07", "模拟器<em>覆盖度</em>", "证明模拟器的充分性")

    if coverage.untriggered_rules:
        with st.expander(
            f"未触发的条件规则 · {len(coverage.untriggered_rules)} 条",
            expanded=False,
        ):
            rows = []
            for r in coverage.untriggered_rules:
                rows.append(
                    f'<div class="rule-item">'
                    f'  <div class="rule-mark na">·</div>'
                    f'  <div>'
                    f'    <div class="rule-text">{r.description}</div>'
                    f'    <div class="rule-meta-row">'
                    f'      <span class="id">{r.rule_id}</span>'
                    f'      {_sev_pill(r.severity)}'
                    f'    </div>'
                    f'    <div class="rule-evidence">{r.evaluation_hint}</div>'
                    f'  </div>'
                    f'</div>'
                )
            st.markdown(f'<div class="rule-list">{"".join(rows)}</div>', unsafe_allow_html=True)

    if coverage.triggered_by_persona:
        with st.expander("各角色触发的条件规则", expanded=False):
            for persona_type, rule_ids in sorted(coverage.triggered_by_persona.items()):
                tags = " ".join(
                    f'<span class="pill pill-brand">{rid}</span>'
                    for rid in sorted(rule_ids)
                )
                st.markdown(
                    f'<div style="margin-bottom:1rem;">'
                    f'  <div class="subhead">{_persona_label(persona_type)}</div>'
                    f'  <div class="tag-cloud">{tags}</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

    _section_head("08", "会话<em>归档</em>", f"共 {total} 个对话")
    for item, report in zip(st.session_state.transcripts, reports):
        _render_session_card(item, report)
