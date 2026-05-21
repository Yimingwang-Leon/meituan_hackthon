"""端到端 smoke test：验证 conditional 规则的两步 judge 真实跑得通。

跑通一次完整链路：
  build_simulation_plan(instruction)
    → 挑 1 个 target 为 conditional 规则的 SimulationCase
    → 跑对话（OutboundSession + UserSimulator）
    → evaluate_session 用 parsed_rules 评测
    → 断言两步 judge 字段（trigger_confidence / compliance_confidence / phase）有值

不写 session 到磁盘，不依赖 UI。失败立刻 exit 1，并打印诊断。

用法：
  python smoke_test_e2e.py            # 默认跑 instructions/cancel.json
  python smoke_test_e2e.py confirm    # 跑 instructions/confirm.json
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

from src.agent import make_outbound_agent
from src.evaluator import evaluate_session
from src.session import OutboundSession
from src.simulation_plan import build_simulation_plan
from src.simulator import UserSimulator
from src.types import SessionMeta

MAX_TURNS = 12


def _die(msg: str) -> None:
    print(f"\n❌ FAIL: {msg}", flush=True)
    sys.exit(1)


def _load_instruction(name: str) -> tuple[str, str]:
    project_root = Path(__file__).resolve().parent
    path = project_root / "instructions" / f"{name}.json"
    if not path.exists():
        _die(f"找不到 {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    instruction = data.get("instruction")
    if not isinstance(instruction, str) or not instruction.strip():
        _die(f"{path} 缺少 instruction 字段")
    return instruction.strip(), name


def main() -> None:
    project_root = Path(__file__).resolve().parent
    load_dotenv(project_root / ".env")

    if not (os.getenv("DEEPSEEK_API_KEY") or os.getenv("OPENAI_API_KEY")):
        _die("缺少 DEEPSEEK_API_KEY 或 OPENAI_API_KEY，请在 agent/.env 中填写")

    name = sys.argv[1] if len(sys.argv) > 1 else "cancel"
    instruction, source_label = _load_instruction(name)

    print(f"\n[1/4] 解析 instruction → SimulationPlan ({source_label})", flush=True)
    plan = build_simulation_plan(instruction, num_sets=1)
    conditional_rules = [r for r in plan.parsed_rules if r.rule_type == "conditional"]
    if not conditional_rules:
        _die(f"{source_label} 解析出来没有 conditional 规则，无法测两步 judge")
    print(
        f"  parsed_rules: {len(plan.parsed_rules)} 条"
        f"（conditional: {len(conditional_rules)}，"
        f"required: {sum(1 for r in plan.parsed_rules if r.rule_type == 'required')}，"
        f"forbidden: {sum(1 for r in plan.parsed_rules if r.rule_type == 'forbidden')}）",
        flush=True,
    )

    sub_plan = plan.sub_plans[0]
    conditional_cases = [
        c for c in sub_plan.test_cases
        if c.test_goal.target_rule_type == "conditional"
    ]
    if not conditional_cases:
        _die("没有 target 为 conditional 的 SimulationCase")
    case = conditional_cases[0]
    print(
        f"  挑中 case: {case.label}"
        f"\n    target_rule_id: {case.target_rule_id}"
        f"\n    rule_description: {case.test_goal.rule_description}",
        flush=True,
    )

    print(f"\n[2/4] 跑对话（最多 {MAX_TURNS} 轮）", flush=True)
    session_id = f"smoke:{source_label}:{case.test_id}"
    outbound = OutboundSession(
        SessionMeta(
            session_id=session_id,
            source_label=source_label,
            instruction_snapshot=sub_plan.filled_instruction,
            scenario_context=sub_plan.placeholder_values,
        ),
        agent=make_outbound_agent(sub_plan.filled_instruction),
    )
    simulator = UserSimulator(
        case, plan.agent_spec, sub_plan.placeholder_values, session_id=session_id
    )

    agent_turn = outbound.start()
    print(f"  Agent: {agent_turn.reply_text}", flush=True)

    for turn_idx in range(MAX_TURNS):
        if agent_turn.should_end:
            break
        user_turn = simulator.reply(agent_turn.reply_text)
        print(f"  User:  {user_turn.reply_text}", flush=True)
        if user_turn.should_end:
            outbound.record_user(user_turn.reply_text)
            break
        agent_turn = outbound.reply(user_turn.reply_text)
        print(f"  Agent: {agent_turn.reply_text}", flush=True)

    archive = outbound.get_archive(
        persona_type=case.profile_type,
        case_type=case.case_type,
        test_case_id=case.test_id,
        target_rule_id=case.target_rule_id,
        target_rule_type=case.test_goal.target_rule_type,
        target_rule_description=case.test_goal.rule_description,
        target_rule_evaluation_hint=case.test_goal.evaluation_hint,
        target_rule_severity=case.test_goal.severity,
        set_id=sub_plan.set_id,
        set_label=sub_plan.label,
    )
    print(f"  对话结束，共 {len(archive.transcript)} 轮", flush=True)

    print(f"\n[3/4] evaluate_session 用 parsed_rules 评测", flush=True)
    report = evaluate_session(
        archive,
        persona_type=case.profile_type,
        rules=plan.parsed_rules,
        n_samples=3,
        set_id=sub_plan.set_id,
        set_label=sub_plan.label,
    )
    print(
        f"  总分 {report.score:.0%}，平均置信度 {report.mean_confidence:.0%}",
        flush=True,
    )

    print(f"\n[4/4] 断言两步 judge 真实跑了", flush=True)
    target = next(
        (rr for rr in report.rule_results if rr.rule_id == case.target_rule_id),
        None,
    )
    if target is None:
        _die(f"评测结果里找不到 target rule {case.target_rule_id}")

    print(f"  target rule {target.rule_id} 结果：{target.result}", flush=True)
    print(
        f"    trigger_confidence  = {target.trigger_confidence}",
        flush=True,
    )
    print(
        f"    compliance_confidence = {target.compliance_confidence}",
        flush=True,
    )
    print(f"    overall confidence  = {target.confidence:.2f}", flush=True)
    print(f"    triggered = {target.triggered}", flush=True)
    print(f"    trigger_turn = {target.trigger_turn}", flush=True)
    print(f"    response_turn = {target.response_turn}", flush=True)
    print(f"    evidence: {target.evidence}", flush=True)
    if target.matched_failure_criteria:
        print(
            f"    matched_failure_criteria: {target.matched_failure_criteria}",
            flush=True,
        )
    if target.suggestion:
        print(f"    suggestion: {target.suggestion}", flush=True)

    # 核心断言：trigger_confidence 必须非空（即两步路径走过）
    if target.trigger_confidence is None:
        _die(
            f"target conditional rule {target.rule_id} 的 trigger_confidence 是 None — "
            "两步 judge 路径没有走通，evaluate_session 可能错误地走到了单步分支"
        )

    # 所有 sample 应该有 phase 字段
    if not target.all_samples:
        _die(f"{target.rule_id} 的 all_samples 是空，无法验证 phase 字段")
    phases = {s.get("phase") for s in target.all_samples if isinstance(s, dict)}
    if "trigger" not in phases:
        _die(
            f"{target.rule_id} 的 all_samples 不含 phase=trigger 的样本，"
            f"实际包含的 phase: {phases}"
        )

    # 如果触发了，应该同时有 compliance 阶段样本
    if target.triggered:
        if target.compliance_confidence is None:
            _die(
                f"{target.rule_id} triggered=True 但 compliance_confidence 是 None"
            )
        if "compliance" not in phases:
            _die(
                f"{target.rule_id} triggered=True 但 all_samples 不含 phase=compliance"
            )

    # 顺便扫一眼其他 conditional 规则的两步字段
    other_cond_results = [
        rr for rr in report.rule_results
        if rr.rule_type == "conditional" and rr.rule_id != case.target_rule_id
    ]
    print(f"\n  其他 {len(other_cond_results)} 条 conditional 规则的两步状态：", flush=True)
    for rr in other_cond_results:
        flag = "✓" if rr.trigger_confidence is not None else "✗"
        print(
            f"    {flag} [{rr.rule_id}] result={rr.result} "
            f"trigger_conf={rr.trigger_confidence} "
            f"compliance_conf={rr.compliance_confidence}",
            flush=True,
        )

    print(f"\n✅ PASS: {source_label} 的两步 conditional judge 端到端真实跑通", flush=True)


if __name__ == "__main__":
    main()
