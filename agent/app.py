from __future__ import annotations

import os
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from html import escape
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
from src.persona import ALL_PERSONAS
from src.session import OutboundSession
from src.simulation_plan import build_simulation_plan
from src.simulator import UserSimulator
from src.types import SessionArchive, SessionMeta

MAX_TURNS = 15
SESSIONS_DIR = Path(__file__).parent / "sessions"
MEMORY_DIR = Path(__file__).parent / "memory"


def _has_llm_api_key() -> bool:
    return bool(os.getenv("DEEPSEEK_API_KEY") or os.getenv("OPENAI_API_KEY"))

PERSONA_LABELS = {
    "cooperative": "配合型",
    "suspicious": "警惕型",
    "impatient": "急躁型",
    "ambiguous": "模糊型",
    "info_missing": "缺信息型",
    "rejector": "拒收型",
    "hostile": "对抗型",
}

st.set_page_config(
    page_title="对话 Agent 评测系统",
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
header[data-testid="stHeader"] {
    background: transparent !important;
    height: 3rem !important;
}
[data-testid="stToolbar"] {
    display: flex !important;
    background: transparent !important;
}
[data-testid="stToolbar"] [data-testid="stToolbarActions"],
[data-testid="stToolbar"] [data-testid="stDecoration"] {
    display: none !important;
}
[data-testid="stSidebar"][aria-expanded="false"] {
    min-width: 336px !important;
    max-width: 336px !important;
    transform: none !important;
    pointer-events: auto !important;
}
[data-testid="stSidebarCollapsedControl"] {
    display: flex !important;
    position: fixed !important;
    top: 0.75rem !important;
    left: 0.75rem !important;
    z-index: 999999 !important;
    width: 44px !important;
    height: 44px !important;
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    box-shadow: 0 4px 14px rgba(15,23,42,0.08) !important;
    align-items: center !important;
    justify-content: center !important;
    pointer-events: auto !important;
}
[data-testid="stSidebarCollapsedControl"] img,
[data-testid="stSidebarCollapsedControl"] [data-testid="stLogoSpacer"] {
    display: none !important;
}
[data-testid="stSidebarCollapsedControl"] button {
    display: inline-flex !important;
    align-items: center !important;
    justify-content: center !important;
    width: 40px !important;
    height: 40px !important;
    color: var(--text) !important;
    background: transparent !important;
    border: none !important;
}
[data-testid="stSidebarCollapsedControl"] svg {
    display: block !important;
    color: var(--text) !important;
    fill: currentColor !important;
}

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

.summary-box {
    margin: 0.75rem 0 1.25rem;
    padding: 1.1rem 1.25rem;
    background: linear-gradient(90deg, var(--brand-soft) 0%, var(--surface) 72%);
    border: 1px solid var(--border);
    border-left: 4px solid var(--brand);
    border-radius: 8px;
    box-shadow: 0 1px 2px rgba(28, 25, 23, 0.04);
}
.summary-title {
    font-size: 1rem;
    font-weight: 700;
    color: var(--text);
    margin-bottom: 0.45rem;
}
.summary-line {
    font-size: 0.925rem;
    line-height: 1.75;
    color: var(--text-2);
}
.summary-line strong {
    color: var(--text);
    font-weight: 700;
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
.rule-mark.pass            { background: var(--success-soft); color: var(--success); border: 1px solid #a7f3d0; }
.rule-mark.fail            { background: var(--danger-soft);  color: var(--danger);  border: 1px solid #fecaca; }
.rule-mark.na              { background: var(--surface-hi);   color: var(--text-4);  border: 1px solid var(--border); }
.rule-mark.trigger-failed  { background: var(--warning-soft); color: var(--warning); border: 1px solid #fed7aa; }
.rule-trigger {
    font-size: 0.78rem;
    color: var(--text-3);
    margin-top: 0.375rem;
    font-family: var(--mono);
}
.rule-text {
    font-size: 0.9rem;
    color: var(--text);
    line-height: 1.5;
    font-weight: 500;
}
.rule-verdict {
    font-size: 0.86rem;
    color: var(--text-2);
    margin-top: 0.4rem;
    line-height: 1.5;
}
.rule-verdict strong {
    color: var(--text);
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
.rule-rationale {
    font-size: 0.82rem;
    color: var(--text-2);
    margin-top: 0.5rem;
    padding: 0.5rem 0.75rem;
    background: var(--surface-hi);
    border: 1px solid var(--border);
    border-radius: 6px;
    line-height: 1.55;
}
.rule-suggestion {
    font-size: 0.82rem;
    color: var(--success);
    margin-top: 0.5rem;
    padding: 0.5rem 0.75rem;
    background: var(--success-soft);
    border-left: 3px solid var(--success);
    border-radius: 0 6px 6px 0;
    line-height: 1.55;
}
.sample-details {
    margin-top: 0.5rem;
    font-size: 0.82rem;
    color: var(--text-2);
}
.sample-details summary {
    cursor: pointer;
    color: var(--text-3);
    font-weight: 600;
    user-select: none;
}
.sample-item {
    margin-top: 0.5rem;
    padding: 0.5rem 0.75rem;
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 6px;
    line-height: 1.5;
}
.sample-meta {
    font-family: var(--mono);
    font-size: 0.72rem;
    color: var(--text-3);
    margin-bottom: 0.25rem;
}
.sample-meta strong {
    color: var(--text-2);
    font-family: var(--sans);
    font-size: 0.78rem;
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
    ("run_timing", {}),
    ("run_complete", False),
]:
    if key not in st.session_state:
        st.session_state[key] = default


# ═══════════════════════════════════════════════════════════════════════════
# 渲染辅助
# ═══════════════════════════════════════════════════════════════════════════
def _score_pill(score: float) -> str:
    if score >= 0.8: return f'<span class="pill pill-success">通过 · {score:.0%}</span>'
    if score >= 0.5: return f'<span class="pill pill-warning">部分通过 · {score:.0%}</span>'
    return f'<span class="pill pill-danger">失败 · {score:.0%}</span>'


def _conf_pill(c: float) -> str:
    if c >= 0.99: return f'<span class="pill pill-success">Judge一致率 {c:.0%}</span>'
    if c >= 0.66: return f'<span class="pill pill-warning">Judge一致率 {c:.0%}</span>'
    return f'<span class="pill pill-danger">Judge一致率 {c:.0%}</span>'


def _sev_pill(sev: str) -> str:
    label = _severity_label(sev)
    cls = {"critical": "pill-danger", "major": "pill-warning", "minor": "pill-neutral"}.get(sev, "pill-neutral")
    return f'<span class="pill {cls}">{label}</span>'


def _severity_label(sev: str) -> str:
    return {"critical": "关键", "major": "重要", "minor": "一般"}.get(sev, sev)


def _persona_label(persona_type: str) -> str:
    return PERSONA_LABELS.get(persona_type, persona_type)


def _now_text() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _format_duration(seconds: object) -> str:
    try:
        total_seconds = max(0.0, float(seconds))
    except (TypeError, ValueError):
        return "-"
    if total_seconds < 60:
        return f"{total_seconds:.1f}s"
    rounded = int(round(total_seconds))
    minutes, sec = divmod(rounded, 60)
    if minutes < 60:
        return f"{minutes}m {sec:02d}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h {minutes:02d}m {sec:02d}s"


def _timing_seconds(timing: dict[str, object], key: str) -> float:
    return float(timing.get(key, 0) or 0)


def _timing_session_sum(timing: dict[str, object], key: str) -> float:
    sessions = timing.get("sessions", [])
    if not isinstance(sessions, list):
        return 0.0
    return sum(float(session.get(key, 0) or 0) for session in sessions)


def _timing_summary_text(timing: dict[str, object]) -> str:
    sessions = timing.get("sessions", [])
    if not isinstance(sessions, list):
        sessions = []
    lines = [
        "评测耗时记录",
        f"开始时间: {timing.get('started_at', '-')}",
        f"结束时间: {timing.get('finished_at', '-')}",
        f"总耗时: {_format_duration(timing.get('total_seconds', 0))}",
        f"规划耗时: {_format_duration(timing.get('plan_seconds', 0))}",
        f"对话生成耗时: {_format_duration(_timing_session_sum(timing, 'dialogue_seconds'))}",
        f"规则评测耗时: {_format_duration(_timing_session_sum(timing, 'eval_seconds'))}",
        f"会话数: {len(sessions)}",
        "",
        "逐会话:",
    ]
    for index, session in enumerate(sessions, start=1):
        lines.append(
            f"{index}. {session.get('session_label', '-')}"
            f" | total={_format_duration(session.get('total_seconds', 0))}"
            f" | dialogue={_format_duration(session.get('dialogue_seconds', 0))}"
            f" | eval={_format_duration(session.get('eval_seconds', 0))}"
        )
    return "\n".join(lines)


def _render_timing_panel(timing: dict[str, object]) -> None:
    if not timing:
        return
    sessions = timing.get("sessions", [])
    session_count = len(sessions) if isinstance(sessions, list) else 0
    total_seconds = _timing_seconds(timing, "total_seconds")
    plan_seconds = _timing_seconds(timing, "plan_seconds")
    dialogue_seconds = _timing_session_sum(timing, "dialogue_seconds")
    eval_seconds = _timing_session_sum(timing, "eval_seconds")
    avg_session_seconds = (
        _timing_session_sum(timing, "total_seconds") / session_count
        if session_count else 0.0
    )

    st.markdown(
        f'<div class="bento">'
        f'{_bento_cell("总耗时", _format_duration(total_seconds), f"{session_count} 个会话", feature=True)}'
        f'{_bento_cell("规划耗时", _format_duration(plan_seconds), "解析规则 + 占位符 + 测试对话")}'
        f'{_bento_cell("对话耗时", _format_duration(dialogue_seconds), "被测 Agent + 用户模拟器")}'
        f'{_bento_cell("评测耗时", _format_duration(eval_seconds), f"平均每会话 {_format_duration(avg_session_seconds)}")}'
        f'</div>',
        unsafe_allow_html=True,
    )
    with st.expander("复制耗时明细", expanded=False):
        st.text_area(
            "耗时明细",
            value=_timing_summary_text(timing),
            height=260,
            label_visibility="collapsed",
        )


def _rule_detail_html(label: str, value: str, css_class: str) -> str:
    value = value.strip()
    if not value:
        return ""
    return (
        f'<div class="{css_class}">'
        f'<strong>{escape(label)}</strong>：{escape(value)}'
        f'</div>'
    )


def _result_label(result: object) -> str:
    return {
        "pass": "通过",
        "fail": "失败",
        "not_applicable": "未触发",
        "trigger_failed": "目标未触发",
        "triggered": "已触发",
        "not_triggered": "未触发",
    }.get(str(result), str(result))


def _result_explanation(result: object) -> str:
    return {
        "pass": "规则已触发，Agent 做到了要求。",
        "fail": "规则已触发，但 Agent 没有做到要求。",
        "not_applicable": "这段对话没有出现规则触发条件，不计入通过率。",
        "trigger_failed": "这是目标测试，但模拟用户没有演出目标场景。",
    }.get(str(result), "")


def _votes_text(votes: dict[str, int]) -> str:
    if not votes:
        return ""
    return " / ".join(
        f"{_result_label(result)}×{count}"
        for result, count in votes.items()
    )


def _session_result_label(report: EvaluationReport) -> str:
    if len(report.rule_results) == 1:
        return _result_label(report.rule_results[0].result)
    if report.score >= 0.8:
        return "通过"
    if report.score >= 0.5:
        return "部分通过"
    return "失败"


def _method_label(evaluated_by: object) -> str:
    return "程序自动检查" if str(evaluated_by) == "deterministic" else "LLM Judge"


def _positive_int(value: object) -> int:
    try:
        parsed = int(value or 0)
    except (TypeError, ValueError):
        return 0
    return parsed if parsed > 0 else 0


def _sample_meta_html(sample: dict[str, object]) -> str:
    phase = str(sample.get("phase", "single"))
    index = escape(str(sample.get("sample_index", "?")))
    result = escape(_result_label(sample.get("result", "")))
    trigger_turn = _positive_int(sample.get("trigger_turn"))
    response_turn = _positive_int(sample.get("response_turn"))

    if phase == "trigger":
        turn_text = f"第 {trigger_turn} 轮" if trigger_turn else "未定位轮次"
        return (
            f'<div class="sample-meta"><strong>触发判定 {index}：{result}</strong>'
            f' · {turn_text}</div>'
        )
    if phase == "compliance":
        turn_text = f"Agent响应第 {response_turn} 轮" if response_turn else "无明确响应轮次"
        return (
            f'<div class="sample-meta"><strong>合规判定 {index}：{result}</strong>'
            f' · {turn_text}</div>'
        )

    parts = [f'<strong>第 {index} 次 Judge：{result}</strong>']
    if trigger_turn:
        parts.append(f"触发第 {trigger_turn} 轮")
    if response_turn:
        parts.append(f"Agent响应第 {response_turn} 轮")
    return f'<div class="sample-meta">{" · ".join(parts)}</div>'


def _samples_html(
    samples: list[dict[str, object]],
    evaluated_by: object = "llm_judge",
) -> str:
    if len(samples) <= 1:
        return ""

    items = []
    for sample in samples:
        evidence = escape(str(sample.get("evidence", "")))
        rationale = escape(str(sample.get("rationale", "")))
        suggestion = escape(str(sample.get("suggestion", "")))
        suggestion_html = (
            f'<div><strong>改进建议</strong>：{suggestion}</div>'
            if suggestion else ""
        )
        items.append(
            f'<div class="sample-item">'
            f'  {_sample_meta_html(sample)}'
            f'  <div><strong>证据</strong>：{evidence}</div>'
            f'  <div><strong>判定依据</strong>：{rationale}</div>'
            f'  {suggestion_html}'
            f'</div>'
        )

    return (
        f'<details class="sample-details">'
        f'  <summary>评测方式：{escape(_method_label(evaluated_by))} · '
        f'展开 {len(samples)} 次明细</summary>'
        f'  {"".join(items)}'
        f'</details>'
    )


def _confidence_detail_html(rule_result: object) -> str:
    trigger_confidence = getattr(rule_result, "trigger_confidence", None)
    if trigger_confidence is None:
        return ""

    parts = [f"触发一致率 {float(trigger_confidence):.0%}"]
    compliance_confidence = getattr(rule_result, "compliance_confidence", None)
    if compliance_confidence is not None:
        parts.append(f"合规一致率 {float(compliance_confidence):.0%}")
    return f'<div class="rule-trigger">{" · ".join(parts)}</div>'


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


def _short_text(text: str, limit: int = 18) -> str:
    text = text.strip()
    if len(text) <= limit:
        return text
    return f"{text[:limit]}..."


def _render_report_summary(
    reports: list[EvaluationReport],
    coverage,
    avg_score: float,
    avg_conf: float,
) -> None:
    rule_results = [rr for report in reports for rr in report.rule_results]
    failed = [rr for rr in rule_results if rr.result == "fail"]
    high_impact_failed = [
        rr for rr in failed
        if rr.severity in ("critical", "major")
    ]
    judge_disagreements = [
        rr for rr in rule_results
        if rr.confidence < 0.99 and rr.result not in ("not_applicable", "trigger_failed")
    ]

    lines = [
        (
            f"本次共运行 <strong>{len(reports)}</strong> 段测试对话，"
            f"规则通过率 <strong>{avg_score:.0%}</strong>，"
            f"Judge一致率 <strong>{avg_conf:.0%}</strong>。"
        )
    ]

    if coverage.primary_attempted:
        lines.append(
            f"目标场景未触发率 <strong>{coverage.trigger_failure_rate:.0%}</strong>，"
            f"<strong>{coverage.trigger_failed_count}/{coverage.primary_attempted}</strong> "
            "个目标场景没有演出来；这部分优先看模拟器或测试场景，不直接算 Agent 失败。"
        )

    if failed:
        top_rules = Counter(
            (rr.rule_id, rr.description)
            for rr in failed
        ).most_common(3)
        top_text = "、".join(
            f"{rule_id} {_short_text(description)}"
            + (f"（{count}次）" if count > 1 else "")
            for (rule_id, description), count in top_rules
        )
        impact_text = (
            f"其中关键/重要失败 <strong>{len(high_impact_failed)}</strong> 条"
            if high_impact_failed
            else "暂无关键/重要规则失败"
        )
        lines.append(
            f"发现 <strong>{len(failed)}</strong> 条规则失败，{impact_text}；"
            f"主要集中在 <strong>{escape(top_text)}</strong>。"
        )
    else:
        lines.append(
            "未发现规则失败；建议抽查 Judge 一致率较低或目标场景未触发的记录。"
        )

    if judge_disagreements:
        lines.append(
            f"有 <strong>{len(judge_disagreements)}</strong> 条规则出现 Judge 判断分歧，"
            "展开明细可以看到每次 Judge 的证据差异。"
        )
    else:
        lines.append("Judge 判断没有明显分歧，整体结果更适合作为交付报告直接展示。")

    line_html = "".join(f'<div class="summary-line">{line}</div>' for line in lines)
    st.markdown(
        f'<div class="summary-box">'
        f'  <div class="summary-title">评测结论摘要</div>'
        f'  {line_html}'
        f'</div>',
        unsafe_allow_html=True,
    )


def _build_heatmap_df(reports: list[EvaluationReport]) -> pd.DataFrame:
    rows = []
    for report in reports:
        for rr in report.rule_results:
            score = 1.0 if rr.result == "pass" else 0.0 if rr.result == "fail" else None
            rows.append({
                "规则": f"{rr.rule_id}  {rr.description[:26]}",
                "rule_id": rr.rule_id,
                "规则等级": _severity_label(rr.severity),
                "用户类型": _persona_label(report.persona_type),
                "score": score,
            })
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    return (
        df.dropna(subset=["score"])
        .groupby(["规则", "rule_id", "规则等级", "用户类型"])
        .agg(通过率=("score", "mean"), 测试对话数=("score", "count"))
        .reset_index()
    )


def _render_heatmap(reports: list[EvaluationReport]) -> None:
    df = _build_heatmap_df(reports)
    if df.empty:
        st.caption("暂无可展示的规则通过情况")
        return

    height = max(280, 30 * df["规则"].nunique())
    chart = (
        alt.Chart(df)
        .mark_rect(stroke="#ffffff", strokeWidth=2, cornerRadius=4)
        .encode(
            x=alt.X("用户类型:N", title=None, axis=alt.Axis(
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
                alt.Tooltip("用户类型:N"),
                alt.Tooltip("规则等级:N"),
                alt.Tooltip("通过率:Q", format=".0%"),
                alt.Tooltip("测试对话数:Q"),
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
        {"用户类型": _persona_label(k), "通过率": sum(v) / len(v)}
        for k, v in by_persona.items()
    ])

    chart = (
        alt.Chart(df)
        .mark_bar(cornerRadiusEnd=4, height=22)
        .encode(
            y=alt.Y("用户类型:N", sort="-x", title=None,
                    axis=alt.Axis(labelPadding=10, ticks=False, domain=False)),
            x=alt.X("通过率:Q", scale=alt.Scale(domain=[0, 1]),
                    axis=alt.Axis(format="%", title=None, grid=True)),
            color=alt.Color(
                "通过率:Q",
                scale=alt.Scale(range=["#dc2626", "#f59e0b", "#059669"], domain=[0, 0.5, 1]),
                legend=None,
            ),
            tooltip=[alt.Tooltip("用户类型:N"), alt.Tooltip("通过率:Q", format=".0%")],
        )
        .properties(height=max(200, 42 * len(df)))
    )
    st.altair_chart(chart, width="stretch")


def _render_session_card(item: dict, report: EvaluationReport) -> None:
    persona_label = _persona_label(report.persona_type)
    session_label = item.get("simulator_label") or persona_label
    session_result = _session_result_label(report)
    timing = item.get("timing") or {}
    timing_label = (
        f"   ·   耗时 {_format_duration(timing.get('total_seconds'))}"
        if timing else ""
    )
    head = (
        f"{session_label}"
        f"   ·   {session_result}"
        f"   ·   Judge一致率 {report.mean_confidence:.0%}"
        f"{timing_label}"
    )

    with st.expander(head, expanded=False):
        scenario_context = item.get("scenario_context") or {}
        if scenario_context:
            context_text = "  ·  ".join(
                f"`{key}` = `{value}`"
                for key, value in scenario_context.items()
            )
            st.caption(f"场景上下文：{context_text}")
        if timing:
            st.caption(
                "耗时："
                f"对话 {_format_duration(timing.get('dialogue_seconds'))} · "
                f"评测 {_format_duration(timing.get('eval_seconds'))} · "
                f"总计 {_format_duration(timing.get('total_seconds'))}"
            )
        col_l, col_r = st.columns([1, 1.15])

        with col_l:
            st.markdown('<div class="subhead">对话记录</div>', unsafe_allow_html=True)
            for entry in item["transcript"]:
                role = "assistant" if entry.speaker == "agent" else "user"
                with st.chat_message(role):
                    st.write(entry.text)

        with col_r:
            st.markdown('<div class="subhead">规则评测</div>', unsafe_allow_html=True)
            target_rule_id = item.get("target_rule_id")
            visible_rule_results = [
                rr for rr in report.rule_results
                if not target_rule_id or rr.rule_id == target_rule_id
            ]
            rows = []
            for rr in visible_rule_results:
                mark_cls = {
                    "pass": "pass",
                    "fail": "fail",
                    "not_applicable": "na",
                    "trigger_failed": "trigger-failed",
                }[rr.result]
                mark_char = {
                    "pass": "✓",
                    "fail": "✗",
                    "not_applicable": "—",
                    "trigger_failed": "⚠",
                }[rr.result]
                votes_text = _votes_text(rr.votes)
                verdict = (
                    f'<div class="rule-verdict">'
                    f'<strong>结论：{_result_label(rr.result)}</strong>'
                    f' · {_result_explanation(rr.result)}'
                    f'{(" · 投票：" + escape(votes_text)) if votes_text else ""}'
                    f'</div>'
                )
                evidence = (
                    _rule_detail_html("证据", rr.evidence, "rule-evidence")
                    if rr.result != "not_applicable" else ""
                )
                rationale = _rule_detail_html(
                    "判定依据",
                    getattr(rr, "rationale", ""),
                    "rule-rationale",
                )
                matched_failure_criteria = getattr(
                    rr,
                    "matched_failure_criteria",
                    [],
                )
                matched = _rule_detail_html(
                    "命中失败标准",
                    "；".join(matched_failure_criteria),
                    "rule-rationale",
                )
                suggestion = (
                    _rule_detail_html(
                        "改进建议",
                        getattr(rr, "suggestion", ""),
                        "rule-suggestion",
                    )
                    if rr.result == "fail" else ""
                )
                samples = _samples_html(
                    getattr(rr, "all_samples", []),
                    getattr(rr, "evaluated_by", "llm_judge"),
                )
                confidence_detail = _confidence_detail_html(rr)
                # 触发信息条
                trigger_info = ""
                if rr.rule_type == "conditional" and rr.triggered is not None:
                    if rr.triggered:
                        trigger_turn = _positive_int(rr.trigger_turn)
                        response_turn = _positive_int(rr.response_turn)
                        response_text = (
                            f" · Agent 响应第 {response_turn} 轮"
                            if response_turn else ""
                        )
                        trigger_info = (
                            f'<div class="rule-trigger">触发于第 {trigger_turn} 轮'
                            f'{response_text}</div>'
                        )
                    elif rr.is_primary:
                        trigger_info = (
                            '<div class="rule-trigger">⚠ 目标规则未触发：模拟用户没有演出目标场景</div>'
                        )
                rows.append(
                    f'<div class="rule-item">'
                    f'  <div class="rule-mark {mark_cls}">{mark_char}</div>'
                    f'  <div>'
                    f'    <div class="rule-text">{escape(rr.description)}</div>'
                    f'    <div class="rule-meta-row">'
                    f'      <span class="id">{escape(rr.rule_id)}</span>'
                    f'      {_sev_pill(rr.severity)}'
                    f'      {_conf_pill(rr.confidence)}'
                    f'    </div>'
                    f'    {confidence_detail}'
                    f'    {verdict}'
                    f'    {trigger_info}'
                    f'    {evidence}'
                    f'    {rationale}'
                    f'    {matched}'
                    f'    {suggestion}'
                    f'    {samples}'
                    f'  </div>'
                    f'</div>'
                )
            if target_rule_id and len(report.rule_results) > len(visible_rule_results):
                st.caption(f"当前卡片仅展示目标规则 `{target_rule_id}` 的评测结果。")
            st.markdown(f'<div class="rule-list">{"".join(rows)}</div>', unsafe_allow_html=True)


@dataclass(frozen=True)
class TestRunResult:
    index: int
    session_label: str
    archive: SessionArchive
    report: EvaluationReport
    item: dict
    timing_entry: dict


def _run_test_case(
    index: int,
    total: int,
    sub_plan,
    case,
    agent_spec,
    target_rule,
    judge_samples: int,
) -> TestRunResult:
    persona_label = _persona_label(case.profile_type)
    session_label = (
        f"{sub_plan.set_id} · {case.target_rule_id} · "
        f"{case.case_type_label} · {persona_label}"
    )
    session_id = f"{sub_plan.set_id}:{case.test_id}"
    session_timer_start = time.perf_counter()
    print(f"\n[{index}/{total}] 开始 {session_label}", flush=True)

    agent = make_outbound_agent(sub_plan.filled_instruction)
    session_meta = SessionMeta(
        session_id=session_id,
        source_label="streamlit",
        instruction_snapshot=sub_plan.filled_instruction,
        scenario_context=sub_plan.placeholder_values,
    )
    outbound = OutboundSession(session_meta, agent=agent)
    simulator = UserSimulator(
        case,
        agent_spec,
        sub_plan.placeholder_values,
        session_id=session_id,
    )

    dialogue_timer_start = time.perf_counter()
    try:
        print(f"  [{index}/{total}] agent.start()", flush=True)
        agent_turn = outbound.start()
        for turn_idx in range(MAX_TURNS):
            if agent_turn.should_end:
                print(f"  [{index}/{total}] agent 结束于第 {turn_idx + 1} 轮", flush=True)
                break
            user_turn = simulator.reply(agent_turn.reply_text)
            if user_turn.should_end:
                outbound.record_user(user_turn.reply_text)
                print(f"  [{index}/{total}] user 结束于第 {turn_idx + 1} 轮", flush=True)
                break
            agent_turn = outbound.reply(user_turn.reply_text)
        else:
            print(f"  [{index}/{total}] 达到 MAX_TURNS={MAX_TURNS}", flush=True)
    except Exception as exc:
        raise RuntimeError(f"对话生成失败：{session_label}") from exc
    dialogue_seconds = time.perf_counter() - dialogue_timer_start

    outbound.save_archive(
        SESSIONS_DIR,
        "agent_end",
        persona_type=case.profile_type,
        case_type=case.case_type,
        simulator_label=session_label,
        test_case_id=case.test_id,
        target_rule_id=case.target_rule_id,
        target_rule_type=case.test_goal.target_rule_type,
        target_rule_description=case.test_goal.rule_description,
        target_rule_evaluation_hint=case.test_goal.evaluation_hint,
        target_rule_severity=case.test_goal.severity,
        set_id=sub_plan.set_id,
        set_label=sub_plan.label,
    )
    archive = outbound.get_archive(
        persona_type=case.profile_type,
        case_type=case.case_type,
        simulator_label=session_label,
        test_case_id=case.test_id,
        target_rule_id=case.target_rule_id,
        target_rule_type=case.test_goal.target_rule_type,
        target_rule_description=case.test_goal.rule_description,
        target_rule_evaluation_hint=case.test_goal.evaluation_hint,
        target_rule_severity=case.test_goal.severity,
        set_id=sub_plan.set_id,
        set_label=sub_plan.label,
    )

    print(
        f"  [{index}/{total}] 开始评测目标规则 {case.target_rule_id} × {judge_samples} 采样",
        flush=True,
    )
    eval_timer_start = time.perf_counter()
    try:
        report = evaluate_session(
            archive,
            case.profile_type,
            rules=[target_rule],
            n_samples=judge_samples,
            max_workers=1,
            set_id=sub_plan.set_id,
            set_label=sub_plan.label,
        )
    except Exception as exc:
        raise RuntimeError(f"规则评测失败：{session_label}") from exc
    eval_seconds = time.perf_counter() - eval_timer_start
    session_seconds = time.perf_counter() - session_timer_start

    print(
        f"  [{index}/{total}] 评测完成，通过率 {report.score:.0%}，"
        f"耗时 {_format_duration(session_seconds)} "
        f"(对话 {_format_duration(dialogue_seconds)} / 评测 {_format_duration(eval_seconds)})",
        flush=True,
    )

    item = {
        "session_id": session_id,
        "persona_type": case.profile_type,
        "case_type": case.case_type,
        "case_type_label": case.case_type_label,
        "simulator_label": session_label,
        "test_case_id": case.test_id,
        "target_rule_id": case.target_rule_id,
        "set_id": sub_plan.set_id,
        "set_label": sub_plan.label,
        "scenario_context": sub_plan.placeholder_values,
        "transcript": archive.transcript,
        "timing": {
            "dialogue_seconds": dialogue_seconds,
            "eval_seconds": eval_seconds,
            "total_seconds": session_seconds,
        },
    }
    timing_entry = {
        "session_id": session_id,
        "session_label": session_label,
        "target_rule_id": case.target_rule_id,
        "persona_type": case.profile_type,
        "case_type": case.case_type,
        "dialogue_seconds": dialogue_seconds,
        "eval_seconds": eval_seconds,
        "total_seconds": session_seconds,
    }
    return TestRunResult(
        index=index,
        session_label=session_label,
        archive=archive,
        report=report,
        item=item,
        timing_entry=timing_entry,
    )


# ═══════════════════════════════════════════════════════════════════════════
# 侧边栏
# ═══════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown(
        '<div class="brand-mark"><span class="dot"></span>评测控制台</div>',
        unsafe_allow_html=True,
    )

    st.markdown("### 用户类型")
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
        "LLM Judge 采样次数",
        options=[1, 3],
        value=1,
        format_func=lambda x: "1 次采样 · 快" if x == 1 else "3 次采样 · 更稳",
        help="1 次适合快速检查；3 次可观察 Judge 分歧，避免 2 次采样平票。",
        key="slider_judge_samples",
    )
    st.caption(
        f"当前每条目标规则会评判 {judge_samples} 次，并保存每次 Judge 明细。"
        "3 次更稳，可观察触发/合规判断分歧。"
    )

    st.markdown("### 占位符取值")
    num_placeholder_sets = st.select_slider(
        "占位符取值组数",
        options=[1, 2, 3],
        value=1,
        format_func=lambda x: (
            f"{x} 组 · 标准" if x == 1
            else f"{x} 组 · 标准+边界" if x == 2
            else f"{x} 组 · 标准+边界+高压"
        ),
        help="为指令里的占位符生成多组填充值，跑多遍以测试鲁棒性。",
        key="slider_placeholder_sets",
    )

    st.markdown("### 并行执行")
    parallel_workers = st.select_slider(
        "并行测试数",
        options=[1, 2, 4, 8, 16],
        value=1,
        format_func=lambda x: f"{x} 路" + (" · 顺序" if x == 1 else ""),
        help="同时运行多少个测试对话。并行度越高越快，但也更容易触发模型接口限流或网络抖动。",
        key="slider_parallel_workers",
    )
    st.caption(f"当前最多同时运行 {parallel_workers} 个测试案例。")

    st.markdown(
        f'<div class="side-stat">'
        f'  <div class="label">运行范围</div>'
        f'  <div class="value">单 Prompt 评测</div>'
        f'  <div class="hint">{len(selected_personas)} 个用户类型筛选，'
        f'{judge_samples} 次 Judge 采样，最多展开 {num_placeholder_sets} 组占位符取值，'
        f'{parallel_workers} 路并行</div>'
        f'</div>',
        unsafe_allow_html=True,
    )


# ═══════════════════════════════════════════════════════════════════════════
# Hero
# ═══════════════════════════════════════════════════════════════════════════
st.markdown(
    '<div class="brand-mark"><span class="dot"></span>对话 Agent 指令遵循评测系统</div>',
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
            粘贴任意任务型对话 Agent 指令，系统自动拆解为原子规则，并为每条规则生成目标驱动的测试对话，
            通过多次 Judge 判断，输出可解释、可量化的评测报告。
        </p>
        <div class="hero-meta">
            <div class="hero-meta-item">
                <div class="label">评估方法</div>
                <div class="value">原子规则分解</div>
            </div>
            <div class="hero-meta-item">
                <div class="label">可靠性</div>
                <div class="value">多次判断投票</div>
            </div>
            <div class="hero-meta-item">
                <div class="label">触发情况</div>
                <div class="value">目标驱动模拟</div>
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ═══════════════════════════════════════════════════════════════════════════
# 01 · 任务指令
# ═══════════════════════════════════════════════════════════════════════════
_section_head("01", "任务<em>指令</em>", "粘贴任意任务型对话 Agent prompt")

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
        disabled=not selected_personas,
        width="stretch",
    )


# ═══════════════════════════════════════════════════════════════════════════
# 运行流程
# ═══════════════════════════════════════════════════════════════════════════
if run_clicked:
    if not _has_llm_api_key():
        st.error("缺少 DEEPSEEK_API_KEY 或 OPENAI_API_KEY，请在 agent/.env 中填写")
        st.stop()

    run_started_at = _now_text()
    run_timer_start = time.perf_counter()
    plan_timer_start = time.perf_counter()
    plan_status = st.empty()
    plan_progress = st.progress(0.0, text="准备生成测试计划...")
    plan_stage_state = {"message": "准备生成测试计划"}
    plan_stage_progress = {
        "1/6": 0.10,
        "2/6": 0.25,
        "3/6": 0.40,
        "4/6": 0.55,
        "5/6": 0.72,
        "6/6": 0.88,
    }

    def _on_plan_progress(message: str) -> None:
        plan_stage_state["message"] = message
        prefix = message.split(" ", 1)[0]
        progress_value = plan_stage_progress.get(prefix, 0.05)
        plan_status.info(message)
        plan_progress.progress(progress_value, text=message)

    try:
        with st.spinner("正在生成测试计划..."):
            simulation_plan = build_simulation_plan(
                instructions,
                num_sets=num_placeholder_sets,
                progress_callback=_on_plan_progress,
            )
    except Exception as exc:
        elapsed = time.perf_counter() - run_timer_start
        failed_stage = plan_stage_state["message"]
        plan_progress.progress(1.0, text=f"{failed_stage} · 失败")
        plan_status.error(f"测试计划生成失败：{failed_stage}")
        st.error(f"{type(exc).__name__}: {exc}")
        st.info(f"本次运行已耗时：{_format_duration(elapsed)}")
        exc_text = str(exc)
        if "超时" in exc_text or "RuleParser" in exc_text:
            st.info(
                "当前规则解析和评估判定使用 deepseek-v4-pro，辅助规划步骤使用 flash。"
                "若指令很长，首次规则解析可能超过默认等待时间；"
                "可以在 agent/.env 设置 LLM_REQUEST_TIMEOUT_SECONDS=180 或更大，"
                "然后重启 Streamlit。"
            )
        st.stop()

    plan_seconds = time.perf_counter() - plan_timer_start
    plan_progress.progress(1.0, text="测试计划生成完成")
    plan_status.success(f"测试计划生成完成 · {_format_duration(plan_seconds)}")
    parsed_rules = simulation_plan.parsed_rules
    agent_spec = simulation_plan.agent_spec
    sub_plans = simulation_plan.sub_plans
    placeholders = simulation_plan.placeholders
    rules_by_id = {rule.rule_id: rule for rule in parsed_rules}

    selected_profile_types = {persona.persona_type for persona in selected_personas}

    # 按 (sub_plan, case) 展平后过滤
    plan_case_pairs = [
        (sp, case)
        for sp in sub_plans
        for case in sp.test_cases
        if case.profile_type in selected_profile_types
    ]
    if not plan_case_pairs:
        st.warning("当前用户类型筛选没有命中任何自动生成的测试对话。")
        st.stop()
    rule_display_order = {
        rule.rule_id: index
        for index, rule in enumerate(parsed_rules)
    }
    sub_plan_display_order = {
        sub_plan.set_id: index
        for index, sub_plan in enumerate(sub_plans)
    }
    case_type_display_order = {
        "normal_trigger": 0,
        "ambiguous_trigger": 1,
        "strong_trigger": 2,
        "adversarial_induction": 3,
        "boundary": 4,
    }
    plan_case_pairs.sort(
        key=lambda pair: (
            rule_display_order.get(pair[1].target_rule_id, 9999),
            sub_plan_display_order.get(pair[0].set_id, 9999),
            case_type_display_order.get(pair[1].case_type, 9999),
            pair[1].profile_type,
            pair[1].test_id,
        )
    )

    st.session_state.parsed_rules = parsed_rules
    st.session_state.sub_plans = sub_plans
    st.session_state.placeholders = placeholders

    n_required = sum(1 for r in parsed_rules if r.rule_type == "required")
    n_cond = sum(1 for r in parsed_rules if r.rule_type == "conditional")
    n_forbid = sum(1 for r in parsed_rules if r.rule_type == "forbidden")

    _section_head("02", "解析<em>规则</em> · 占位符取值", "rule_parser + placeholder_extractor + test case generator")
    st.markdown(
        f'<div class="bento">'
        f'{_bento_cell("规则总数", str(len(parsed_rules)), "原子可独立验证", feature=True)}'
        f'{_bento_cell("占位符", str(len(placeholders)), "自动识别")}'
        f'{_bento_cell("取值组", str(len(sub_plans)), "每组一套占位符填值")}'
        f'{_bento_cell("测试对话数", str(len(plan_case_pairs)), "取值组 × 目标规则场景")}'
        f'</div>',
        unsafe_allow_html=True,
    )

    # 规则质量校验报告
    validation_issues = simulation_plan.validation_issues
    if validation_issues:
        errors = [i for i in validation_issues if i.level == "error"]
        warnings = [i for i in validation_issues if i.level == "warning"]
        header = f"⚠️ 规则质量校验：{len(errors)} 错 · {len(warnings)} 警"
        with st.expander(header, expanded=bool(errors)):
            st.caption(
                "💡 规则质量校验在 rule_parser 后自动跑。"
                "Error 触发自动修复（重新调用 LLM 补齐），warning 仅提示。"
            )
            for issue in validation_issues:
                badge = "🔴 ERROR" if issue.level == "error" else "🟡 WARN"
                st.markdown(
                    f"{badge} · **{issue.rule_id}** · `{issue.field}` — {issue.message}"
                )
                if issue.detail:
                    st.caption(f"↳ {issue.detail[:160]}")
    else:
        st.success(f"✅ {len(parsed_rules)} 条规则全部通过质量校验")

    # 显示占位符取值组概览
    if placeholders:
        with st.expander(f"📋 {len(sub_plans)} 组占位符取值明细", expanded=False):
            for sp in sub_plans:
                values_str = "  ·  ".join(f"`{k}` = `{v}`" for k, v in sp.placeholder_values.items())
                st.markdown(
                    f"**{sp.set_id} · {sp.label}**　{sp.scenario_hint}  \n{values_str}"
                )

    st.session_state.reports = []
    st.session_state.transcripts = []
    st.session_state.run_timing = {
        "started_at": run_started_at,
        "finished_at": "",
        "plan_seconds": plan_seconds,
        "total_seconds": 0.0,
        "sessions": [],
    }
    st.session_state.run_complete = False

    total = len(plan_case_pairs)
    effective_parallel_workers = min(parallel_workers, total)

    _section_head(
        "03",
        "实时<em>执行</em>",
        f"{total} 个会话 · {effective_parallel_workers} 路并行",
    )

    progress = st.progress(
        0,
        text=f"准备运行 {total} 个会话 · 并行度 {effective_parallel_workers}",
    )
    live = st.container()
    with live:
        live_slots = [
            st.empty()
            for _ in plan_case_pairs
        ]

    done = 0
    completed_results: dict[int, TestRunResult] = {}
    executor = ThreadPoolExecutor(max_workers=effective_parallel_workers)
    executor_closed = False
    futures = []
    try:
        for index, (sub_plan, case) in enumerate(plan_case_pairs, start=1):
            target_rule = rules_by_id[case.target_rule_id]
            futures.append(
                executor.submit(
                    _run_test_case,
                    index,
                    total,
                    sub_plan,
                    case,
                    agent_spec,
                    target_rule,
                    judge_samples,
                )
            )

        for future in as_completed(futures):
            try:
                result = future.result()
            except Exception as exc:
                for pending in futures:
                    pending.cancel()
                executor.shutdown(wait=False, cancel_futures=True)
                executor_closed = True
                progress.progress(
                    done / total,
                    text=f"并行执行失败 · 已完成 {done}/{total}",
                )
                st.error("测试执行失败，已取消尚未开始的任务。")
                st.error(f"{type(exc).__name__}: {exc}")
                root = exc.__cause__
                if root is not None:
                    st.error(f"底层错误：{type(root).__name__}: {root}")
                st.info(
                    "并行度较高时更容易触发模型接口限流、连接抖动或结构化输出异常；"
                    "可以降低并行度后重新运行。"
                )
                st.stop()

            append_evaluation_memory(result.archive, result.report, MEMORY_DIR)
            completed_results[result.index] = result
            ordered_results = [
                completed_results[index]
                for index in sorted(completed_results)
            ]
            st.session_state.run_timing["sessions"] = [
                ordered_result.timing_entry
                for ordered_result in ordered_results
            ]
            st.session_state.reports = [
                ordered_result.report
                for ordered_result in ordered_results
            ]
            st.session_state.transcripts = [
                ordered_result.item
                for ordered_result in ordered_results
            ]
            with live_slots[result.index - 1].container():
                _render_session_card(result.item, result.report)

            done += 1
            progress.progress(
                done / total,
                text=(
                    f"已完成 · {result.session_label}  ({done}/{total}) · "
                    f"并行度 {effective_parallel_workers} · 按规则顺序展示"
                ),
            )
    finally:
        if not executor_closed:
            executor.shutdown(wait=True, cancel_futures=True)

    progress.progress(1.0, text="评测完成 · 100%")
    st.session_state.run_timing["finished_at"] = _now_text()
    st.session_state.run_timing["total_seconds"] = time.perf_counter() - run_timer_start
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
    _render_report_summary(reports, coverage, avg_score, avg_conf)

    # 主指标行
    st.markdown(
        f'<div class="bento">'
        f'{_bento_cell("规则通过率", f"{avg_score:.0%}", f"共 {total} 个会话平均", feature=True)}'
        f'{_bento_cell("条件规则触发率", f"{coverage.coverage_rate:.0%}", f"{coverage.triggered_conditional}/{coverage.total_conditional} 条条件规则出现")}'
        f'{_bento_cell("Judge一致率", f"{avg_conf:.0%}", "多次 Judge 判断的一致程度")}'
        f'{_bento_cell("测试对话数", f"{total}", "本次实际运行的对话数")}'
        f'</div>',
        unsafe_allow_html=True,
    )
    _render_timing_panel(st.session_state.get("run_timing", {}))

    # 模拟器质量行
    trigger_fail_label = (
        f"{coverage.trigger_failed_count}/{coverage.primary_attempted} 个目标场景未演出"
        if coverage.primary_attempted else "无目标测试"
    )
    st.markdown(
        f'<div class="bento">'
        f'{_bento_cell("目标场景未触发率", f"{coverage.trigger_failure_rate:.0%}", trigger_fail_label, feature=True)}'
        f'{_bento_cell("目标场景测试数", f"{coverage.primary_attempted}", "指派给具体规则的测试对话数")}'
        f'{_bento_cell("目标场景未触发数", f"{coverage.trigger_failed_count}", "模拟用户没有演出目标场景")}'
        f'{_bento_cell("触发过的条件规则", f"{coverage.triggered_conditional}", "至少出现过一次触发条件")}'
        f'</div>',
        unsafe_allow_html=True,
    )
    st.caption(
        "ℹ️ **目标场景未触发率** 反映用户模拟器没有演出目标场景的比例；"
        "它不是 Agent 失败率，数字越低说明测试对话更准确。"
    )

    # Evaluator 类型分布行
    det_count = sum(
        1 for r in parsed_rules if r.checks
    )
    llm_count = len(parsed_rules) - det_count
    det_rate = det_count / len(parsed_rules) if parsed_rules else 0
    det_label = "无确定性规则" if det_count == 0 else f"{det_count}/{len(parsed_rules)} 条规则走确定性检查"
    st.markdown(
        f'<div class="bento">'
        f'{_bento_cell("确定性检查占比", f"{det_rate:.0%}", det_label, feature=True)}'
        f'{_bento_cell("确定性规则", f"{det_count}", "字数 / 关键词 / PII 等可代码判断")}'
        f'{_bento_cell("语义Judge规则", f"{llm_count}", "需要 LLM 判断语义是否合规")}'
        f'{_bento_cell("预计节省Judge调用", f"{det_count * total}", "估算 = 确定性规则数 × 测试对话数")}'
        f'</div>',
        unsafe_allow_html=True,
    )
    st.caption(
        "ℹ️ **Hybrid Evaluator**：确定性规则（字数、关键词、PII）用代码 100% 准确判定，"
        "语义规则才走 LLM Judge + 多次采样投票。"
    )

    _section_head("05", "规则表现 · <em>按用户类型</em>", "红=容易失败 / 绿=稳定通过")
    _render_heatmap(reports)

    _section_head("06", "<em>用户类型</em>通过率对比", "哪类用户更容易暴露问题")
    _render_persona_bar(reports)

    _section_head("07", "模拟器<em>触发情况</em>", "区分 Agent 失败和场景未触发")

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
                    f'    <div class="rule-evidence">触发：{r.trigger_condition or "（全程适用）"}<br/>期望：{r.expected_behavior}</div>'
                    f'  </div>'
                    f'</div>'
                )
            st.markdown(f'<div class="rule-list">{"".join(rows)}</div>', unsafe_allow_html=True)

    if coverage.triggered_by_persona:
        with st.expander("各用户类型触发过的条件规则", expanded=False):
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

    _section_head("08", "逐条<em>测试记录</em>", f"共 {total} 个对话")
    for item, report in zip(st.session_state.transcripts, reports):
        _render_session_card(item, report)
