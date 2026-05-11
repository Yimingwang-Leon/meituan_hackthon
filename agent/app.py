from __future__ import annotations

import os
import sys
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent))
load_dotenv(Path(__file__).parent / ".env")

from src.agent import OUTBOUND_INSTRUCTIONS, make_outbound_agent
from src.evaluator import EvaluationReport, evaluate_session
from src.orders import load_pending_orders
from src.persona import ALL_PERSONAS
from src.rule_parser import parse_rules
from src.session import OutboundSession
from src.simulator import UserSimulator

MAX_TURNS = 15
SESSIONS_DIR = Path(__file__).parent / "sessions"
ORDERS_DIR = Path(__file__).parent / "orders"

st.set_page_config(page_title="美团外呼评测系统", layout="wide")
st.title("美团外呼数字人评测系统")

if "reports" not in st.session_state:
    st.session_state.reports = []
if "transcripts" not in st.session_state:
    st.session_state.transcripts = []

tab_input, tab_conv, tab_report = st.tabs(["输入", "对话记录", "评测报告"])


# ── 输入页 ──────────────────────────────────────────────────────────────────
with tab_input:
    instructions = st.text_area("任务指令", value=OUTBOUND_INSTRUCTIONS, height=380)

    all_orders = load_pending_orders(ORDERS_DIR)
    order_options = [o.order.order_id for o in all_orders]
    selected_order_ids = st.multiselect("选择订单", order_options, default=order_options)

    st.subheader("选择 Persona")
    cols = st.columns(3)
    selected_personas = []
    for i, persona in enumerate(ALL_PERSONAS):
        with cols[i % 3]:
            checked = st.checkbox(persona.persona_type, value=True, key=f"p_{persona.persona_type}")
            if checked:
                selected_personas.append(persona)

    if st.button("开始评测", type="primary", disabled=not selected_personas or not selected_order_ids):
        if not os.getenv("OPENAI_API_KEY"):
            st.error("缺少 OPENAI_API_KEY，请在 agent/.env 中填写")
            st.stop()

        orders = [o for o in all_orders if o.order.order_id in selected_order_ids]

        with st.spinner("解析规则中..."):
            parsed_rules = parse_rules(instructions)
        st.info(f"已解析 {len(parsed_rules)} 条规则")

        agent = make_outbound_agent(instructions)
        st.session_state.reports = []
        st.session_state.transcripts = []

        total = len(orders) * len(selected_personas)
        progress = st.progress(0, text="准备中...")
        done = 0
        results_container = st.container()

        for order in orders:
            for persona in selected_personas:
                label = f"{order.order.order_id} × {persona.persona_type}"
                progress.progress(done / total, text=f"运行中：{label}")

                outbound = OutboundSession(order, agent=agent)
                simulator = UserSimulator(order, persona)
                agent_turn = outbound.start()

                for _ in range(MAX_TURNS):
                    if agent_turn.should_end:
                        break
                    user_turn = simulator.reply(agent_turn.reply_text)
                    if user_turn.should_end:
                        outbound.record_user(user_turn.reply_text)
                        break
                    agent_turn = outbound.reply(user_turn.reply_text)

                outbound.save_archive(SESSIONS_DIR, "agent_end", persona_type=persona.persona_type)
                archive = outbound.get_archive(persona_type=persona.persona_type)
                report = evaluate_session(archive, persona.persona_type, rules=parsed_rules)

                st.session_state.reports.append(report)
                st.session_state.transcripts.append({
                    "order_id": order.order.order_id,
                    "persona_type": persona.persona_type,
                    "transcript": archive.transcript,
                })

                with results_container:
                    icon = "✅" if report.score >= 0.8 else "⚠️" if report.score >= 0.5 else "❌"
                    with st.expander(f"{icon} {label}  {report.score:.0%}", expanded=False):
                        col_left, col_right = st.columns(2)
                        with col_left:
                            st.caption("对话记录")
                            for entry in archive.transcript:
                                if entry.speaker == "agent":
                                    with st.chat_message("assistant"):
                                        st.write(entry.text)
                                else:
                                    with st.chat_message("user"):
                                        st.write(entry.text)
                        with col_right:
                            st.caption("规则评测")
                            for rr in report.rule_results:
                                rule_icon = {"pass": "✅", "fail": "❌", "not_applicable": "➖"}[rr.result]
                                st.markdown(f"{rule_icon} **[{rr.rule_id}/{rr.severity}]** {rr.description}")
                                if rr.result != "not_applicable":
                                    st.caption(rr.evidence)

                done += 1

        progress.progress(1.0, text="评测完成")
        st.success(f"完成！共跑 {total} 个 session。")


# ── 对话记录页 ───────────────────────────────────────────────────────────────
with tab_conv:
    if not st.session_state.transcripts:
        st.info("运行评测后在此查看对话记录。")
    else:
        for item in st.session_state.transcripts:
            label = f"{item['order_id']} — {item['persona_type']}"
            with st.expander(label):
                for entry in item["transcript"]:
                    if entry.speaker == "agent":
                        with st.chat_message("assistant"):
                            st.write(entry.text)
                    else:
                        with st.chat_message("user"):
                            st.write(entry.text)


# ── 评测报告页 ───────────────────────────────────────────────────────────────
with tab_report:
    reports: list[EvaluationReport] = st.session_state.reports

    if not reports:
        st.info("运行评测后在此查看报告。")
    else:
        overall = sum(r.score for r in reports) / len(reports)
        st.metric("总体得分", f"{overall:.0%}")

        st.subheader("各 Persona 得分")
        by_persona: dict[str, list[float]] = {}
        for r in reports:
            by_persona.setdefault(r.persona_type, []).append(r.score)
        persona_avg = {k: sum(v) / len(v) for k, v in by_persona.items()}
        st.bar_chart(persona_avg)

        st.subheader("规则明细")
        for report in reports:
            with st.expander(f"{report.order_id} — {report.persona_type}  {report.score:.0%}"):
                for rr in report.rule_results:
                    icon = {"pass": "✅", "fail": "❌", "not_applicable": "➖"}[rr.result]
                    st.markdown(f"{icon} **[{rr.rule_id}]** {rr.description}")
                    if rr.result != "not_applicable":
                        st.caption(rr.evidence)
