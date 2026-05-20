from __future__ import annotations

import json
import os
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Agent:
    name: str
    instructions: str
    model: str
    output_type: type | None = None


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
        raw_text = _chat_completion(agent, messages)
        final_output = _parse_output(agent, raw_text)
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

    client = OpenAI(
        api_key=_get_api_key(),
        base_url=_get_base_url(),
    )
    response = client.chat.completions.create(
        model=agent.model,
        messages=_build_request_messages(agent, messages),
        stream=False,
        reasoning_effort=os.getenv("DEEPSEEK_REASONING_EFFORT", "high"),
        extra_body={"thinking": {"type": "enabled"}},
    )

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
            f"返回 JSON 必须满足这个 schema：\n{_schema_text(agent.output_type)}"
        )

    return [{"role": "system", "content": system_prompt}, *messages]


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

    payload = _extract_json_payload(raw_text)
    output_type = agent.output_type

    if hasattr(output_type, "model_validate"):
        return output_type.model_validate(payload)
    return payload


def _extract_json_payload(raw_text: str) -> Any:
    cleaned = raw_text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        if "\n" in cleaned:
            cleaned = cleaned.split("\n", 1)[1]
    cleaned = cleaned.strip()
    if cleaned.endswith("```"):
        cleaned = cleaned[:-3].strip()

    for start_char, end_char in (("{", "}"), ("[", "]")):
        start = cleaned.find(start_char)
        end = cleaned.rfind(end_char)
        if start != -1 and end != -1 and end > start:
            candidate = cleaned[start : end + 1]
            try:
                return json.loads(candidate)
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
