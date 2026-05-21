from __future__ import annotations

import json
import os
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Agent:
    name: str
    instructions: str
    model: str
    output_type: type | None = None
    request_timeout_seconds: float | None = None
    max_retries: int | None = None
    parse_max_retries: int | None = None


@dataclass
class _RunResult:
    final_output: Any
    _input_messages: list[dict[str, str]]
    _assistant_text: str

    def to_input_list(self) -> list[dict[str, str]]:
        return [
            *self._input_messages,
            {"role": "assistant", "content": self._assistant_text},
        ]


class Runner:
    @staticmethod
    def run_sync(agent: Agent, input_data: str | list[dict[str, str]]) -> _RunResult:
        messages = _normalize_messages(input_data)
        attempt_messages = messages
        max_parse_retries = (
            _resolve_parse_max_retries(agent) if agent.output_type is not None else 0
        )
        raw_text = ""
        for attempt in range(max_parse_retries + 1):
            raw_text = _chat_completion(agent, attempt_messages)
            try:
                final_output = _parse_output(agent, raw_text)
                break
            except Exception as exc:
                if attempt >= max_parse_retries:
                    raise
                print(
                    f"  {agent.name} 输出结构化 JSON 校验失败，"
                    f"重试修复 {attempt + 1}/{max_parse_retries}："
                    f"{_root_error_summary(exc)}",
                    flush=True,
                )
                attempt_messages = [
                    *messages,
                    {"role": "assistant", "content": raw_text},
                    {
                        "role": "user",
                        "content": _build_json_repair_prompt(agent, raw_text, exc),
                    },
                ]
        else:
            raise RuntimeError(f"{agent.name} 未返回可解析内容")

        return _RunResult(
            final_output=final_output,
            _input_messages=messages,
            _assistant_text=raw_text,
        )


@contextmanager
def trace(*args: Any, **kwargs: Any):
    yield


def _chat_completion(agent: Agent, messages: list[dict[str, str]]) -> str:
    from openai import OpenAI

    request_timeout = _resolve_request_timeout(agent)
    client = OpenAI(
        api_key=_get_api_key(),
        base_url=_get_base_url(),
        timeout=request_timeout,
    )
    request_kwargs: dict[str, Any] = {
        "model": agent.model,
        "messages": _build_request_messages(agent, messages),
        "stream": False,
        "reasoning_effort": _get_reasoning_effort(),
    }
    extra_body = _get_extra_body()
    if extra_body:
        request_kwargs["extra_body"] = extra_body

    max_retries = _resolve_max_retries(agent)
    last_exc: Exception | None = None
    total_attempts = max_retries + 1
    for attempt in range(max_retries + 1):
        attempt_no = attempt + 1
        attempt_start = time.perf_counter()
        timeout_logged = threading.Event()

        def _log_request_timeout() -> None:
            if timeout_logged.is_set():
                return
            timeout_logged.set()
            print(
                f"  {agent.name} 调用 {agent.model} 等待超过 "
                f"{request_timeout:.0f}s，尚未收到模型响应"
                f"（第 {attempt_no}/{total_attempts} 次请求）",
                flush=True,
            )

        timeout_timer = threading.Timer(request_timeout, _log_request_timeout)
        timeout_timer.daemon = True
        timeout_timer.start()
        try:
            response = client.chat.completions.create(**request_kwargs)
            break
        except Exception as exc:
            last_exc = exc
            elapsed = time.perf_counter() - attempt_start
            if (
                (_looks_like_timeout(exc) or elapsed >= request_timeout * 0.95)
                and not timeout_logged.is_set()
            ):
                timeout_logged.set()
                print(
                    f"  {agent.name} 调用 {agent.model} 请求超时："
                    f"等待 {_format_seconds(elapsed)} 后仍未收到模型响应"
                    f"（单次超时 {request_timeout:.0f}s，"
                    f"第 {attempt_no}/{total_attempts} 次请求）",
                    flush=True,
                )
            if attempt >= max_retries:
                retry_text = f"已用完 {max_retries} 次重试机会。" if max_retries else "未开启重试。"
                raise RuntimeError(
                    f"{agent.name} 调用 {agent.model} 失败或超时。"
                    f"当前单次请求超时 {request_timeout:.0f}s，{retry_text}"
                    "如果是 SSL EOF / Connection error，通常是网络或服务端连接抖动；"
                    "如果卡在规则解析/测试计划生成，通常是指令较长或当前模型响应较慢。"
                    "可在 agent/.env 设置 LLM_MAX_RETRIES=2、"
                    "LLM_RETRY_DELAY_SECONDS=1、LLM_REQUEST_TIMEOUT_SECONDS=180。"
                    f"底层错误：{_root_error_summary(exc)}"
                ) from exc

            delay = _get_retry_delay_seconds() * (2 ** attempt)
            print(
                f"  {agent.name} 调用 {agent.model} 失败，"
                f"{delay:.1f}s 后使用第 {attempt + 1}/{max_retries} 次重试机会："
                f"{_root_error_summary(exc)}",
                flush=True,
            )
            time.sleep(delay)
        finally:
            timeout_logged.set()
            timeout_timer.cancel()
    else:
        raise RuntimeError(f"{agent.name} 调用 {agent.model} 未返回结果") from last_exc

    content = response.choices[0].message.content
    if content is None:
        raise RuntimeError(f"{agent.name} 未返回可解析内容")
    return content.strip()


def _build_request_messages(
    agent: Agent,
    messages: list[dict[str, str]],
) -> list[dict[str, str]]:
    system_prompt = agent.instructions
    if agent.output_type is not None:
        system_prompt = (
            f"{agent.instructions}\n\n"
            "你必须只输出合法 JSON，不要输出 Markdown、解释或额外文本。\n"
            "这是运行时结构化输出协议：即使业务指令要求自然口语或不要输出 JSON，"
            "也必须把要展示给用户的自然语言放进 JSON 字段里，不能直接裸输出。\n"
            f"返回 JSON 必须满足这个 schema：\n{_schema_text(agent.output_type)}"
        )

    return [{"role": "system", "content": system_prompt}, *messages]


def _build_json_repair_prompt(agent: Agent, raw_text: str, exc: Exception) -> str:
    schema = _schema_text(agent.output_type) if agent.output_type is not None else "{}"
    return (
        "上一次输出没有通过结构化校验，不能返回列表、空数组、解释文字或 Markdown。\n"
        "请只根据原任务重新输出一个合法 JSON 对象，字段必须满足下面 schema。\n\n"
        f"校验错误：{_root_error_summary(exc)}\n\n"
        f"上一次输出：\n{raw_text[:1200]}\n\n"
        f"必须满足的 schema：\n{schema}"
    )


def _normalize_messages(input_data: str | list[dict[str, str]]) -> list[dict[str, str]]:
    if isinstance(input_data, str):
        return [{"role": "user", "content": input_data}]

    normalized: list[dict[str, str]] = []
    for item in input_data:
        role = str(item.get("role", "user"))
        content = str(item.get("content", ""))
        normalized.append({"role": role, "content": content})
    return normalized


def _parse_output(agent: Agent, raw_text: str) -> Any:
    if agent.output_type is None:
        return raw_text

    output_type = agent.output_type
    try:
        payload = _extract_json_payload(raw_text)
    except RuntimeError:
        fallback_output = _coerce_plain_reply_output(output_type, raw_text)
        if fallback_output is not None:
            return fallback_output
        raise

    if hasattr(output_type, "model_validate"):
        return output_type.model_validate(payload)
    return payload


def _coerce_plain_reply_output(output_type: type, raw_text: str) -> Any | None:
    """Allow dialogue-turn agents to recover when a model returns plain speech."""
    field_names = _model_field_names(output_type)
    if not {"reply_text", "should_end"}.issubset(field_names):
        return None

    # If the model attempted JSON and malformed it, fail loudly instead of guessing.
    if "{" in raw_text or "[" in raw_text:
        return None

    reply_text = _clean_plain_reply(raw_text)
    if not reply_text:
        return None

    should_end = _looks_like_end_reply(reply_text)
    payload: dict[str, Any] = {
        "reply_text": reply_text,
        "should_end": should_end,
    }
    if "end_reason" in field_names:
        payload["end_reason"] = "自然结束" if should_end else None

    if hasattr(output_type, "model_validate"):
        return output_type.model_validate(payload)
    return payload


def _model_field_names(output_type: type) -> set[str]:
    if hasattr(output_type, "model_fields"):
        return set(output_type.model_fields)
    if hasattr(output_type, "__fields__"):
        return set(output_type.__fields__)
    return set()


def _clean_plain_reply(raw_text: str) -> str:
    cleaned = raw_text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`").strip()
        if "\n" in cleaned:
            cleaned = cleaned.split("\n", 1)[1].strip()
    cleaned = cleaned.splitlines()[0].strip()
    for prefix in ("用户：", "User:", "回复：", "reply_text:", "reply_text："):
        if cleaned.startswith(prefix):
            cleaned = cleaned[len(prefix):].strip()
    return cleaned.strip("\"'“”")


def _looks_like_end_reply(reply_text: str) -> bool:
    end_markers = (
        "不用了",
        "不需要了",
        "没事了",
        "就这样",
        "先这样",
        "挂了",
        "再见",
        "拜拜",
        "结束",
        "不聊了",
    )
    return any(marker in reply_text for marker in end_markers)


def _extract_json_payload(raw_text: str) -> Any:
    cleaned = raw_text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        if "\n" in cleaned:
            cleaned = cleaned.split("\n", 1)[1]
    cleaned = cleaned.strip()
    if cleaned.endswith("```"):
        cleaned = cleaned[:-3].strip()

    decoder = json.JSONDecoder()
    for expected_start in ("{", "["):
        for index, char in enumerate(cleaned):
            if char != expected_start:
                continue
            try:
                payload, _ = decoder.raw_decode(cleaned[index:])
                return payload
            except json.JSONDecodeError:
                continue

    raise RuntimeError(f"模型输出不是合法 JSON：{raw_text[:400]}")


def _schema_text(output_type: type) -> str:
    if hasattr(output_type, "model_json_schema"):
        return json.dumps(output_type.model_json_schema(), ensure_ascii=False, indent=2)
    return json.dumps({"type": "object"}, ensure_ascii=False)


def _get_api_key() -> str:
    api_key = os.getenv("DEEPSEEK_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("缺少 DEEPSEEK_API_KEY 或 OPENAI_API_KEY")
    return api_key


def _get_base_url() -> str:
    return (
        os.getenv("DEEPSEEK_BASE_URL")
        or os.getenv("OPENAI_BASE_URL")
        or "https://api.deepseek.com"
    )


def _get_request_timeout() -> float:
    raw_timeout = (
        os.getenv("DEEPSEEK_TIMEOUT_SECONDS")
        or os.getenv("OPENAI_TIMEOUT_SECONDS")
        or os.getenv("LLM_REQUEST_TIMEOUT_SECONDS")
        or "180"
    )
    try:
        timeout = float(raw_timeout)
    except ValueError:
        timeout = 180.0
    return timeout if timeout > 0 else 180.0


def _resolve_request_timeout(agent: Agent) -> float:
    if agent.request_timeout_seconds is None:
        return _get_request_timeout()
    return agent.request_timeout_seconds if agent.request_timeout_seconds > 0 else 180.0


def _get_max_retries() -> int:
    raw_retries = (
        os.getenv("DEEPSEEK_MAX_RETRIES")
        or os.getenv("OPENAI_MAX_RETRIES")
        or os.getenv("LLM_MAX_RETRIES")
        or "2"
    )
    try:
        retries = int(raw_retries)
    except ValueError:
        retries = 2
    return max(0, retries)


def _resolve_max_retries(agent: Agent) -> int:
    if agent.max_retries is None:
        return _get_max_retries()
    return max(0, agent.max_retries)


def _get_parse_max_retries() -> int:
    raw_retries = (
        os.getenv("DEEPSEEK_PARSE_MAX_RETRIES")
        or os.getenv("OPENAI_PARSE_MAX_RETRIES")
        or os.getenv("LLM_PARSE_MAX_RETRIES")
        or "1"
    )
    try:
        retries = int(raw_retries)
    except ValueError:
        retries = 1
    return max(0, retries)


def _resolve_parse_max_retries(agent: Agent) -> int:
    if agent.parse_max_retries is None:
        return _get_parse_max_retries()
    return max(0, agent.parse_max_retries)


def _get_retry_delay_seconds() -> float:
    raw_delay = (
        os.getenv("DEEPSEEK_RETRY_DELAY_SECONDS")
        or os.getenv("OPENAI_RETRY_DELAY_SECONDS")
        or os.getenv("LLM_RETRY_DELAY_SECONDS")
        or "1"
    )
    try:
        delay = float(raw_delay)
    except ValueError:
        delay = 1.0
    return delay if delay >= 0 else 1.0


def _root_error_summary(exc: BaseException) -> str:
    current: BaseException = exc
    seen: set[int] = set()
    while current.__cause__ is not None and id(current.__cause__) not in seen:
        seen.add(id(current))
        current = current.__cause__
    message = str(current).strip()
    return f"{type(current).__name__}: {message}" if message else type(current).__name__


def _looks_like_timeout(exc: BaseException) -> bool:
    text = _root_error_summary(exc).lower()
    return any(marker in text for marker in ("timeout", "timed out", "readtimeout"))


def _format_seconds(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes = int(seconds // 60)
    remainder = seconds - minutes * 60
    return f"{minutes}m {remainder:.0f}s"


def _get_reasoning_effort() -> str:
    return (
        os.getenv("DEEPSEEK_REASONING_EFFORT")
        or os.getenv("OPENAI_REASONING_EFFORT")
        or "low"
    )


def _get_extra_body() -> dict[str, object]:
    raw_thinking = (
        os.getenv("DEEPSEEK_THINKING_ENABLED")
        or os.getenv("DEEPSEEK_THINKING")
        or ""
    ).strip().lower()
    if raw_thinking in {"1", "true", "yes", "on", "enabled"}:
        return {"thinking": {"type": "enabled"}}
    return {}


__all__ = [
    "Agent",
    "Runner",
    "trace",
]
