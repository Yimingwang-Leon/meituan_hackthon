from __future__ import annotations

import unittest
from unittest.mock import patch

from agents import Agent, Runner, _parse_output
from pydantic import BaseModel, Field


class ReplyOutput(BaseModel):
    reply_text: str = Field(description="自然回复")
    should_end: bool = Field(description="是否结束")
    end_reason: str | None = None


class StructuredOutput(BaseModel):
    answer: int


class AgentParsingTest(unittest.TestCase):
    def test_plain_reply_is_coerced_for_dialogue_output(self) -> None:
        agent = Agent(
            name="dialogue",
            instructions="reply",
            model="test",
            output_type=ReplyOutput,
        )

        output = _parse_output(agent, "下午两点到四点。")

        self.assertIsInstance(output, ReplyOutput)
        self.assertEqual(output.reply_text, "下午两点到四点。")
        self.assertFalse(output.should_end)

    def test_plain_ending_reply_sets_should_end(self) -> None:
        agent = Agent(
            name="dialogue",
            instructions="reply",
            model="test",
            output_type=ReplyOutput,
        )

        output = _parse_output(agent, "不用了，先挂了。")

        self.assertTrue(output.should_end)
        self.assertEqual(output.end_reason, "自然结束")

    def test_plain_text_still_fails_for_non_dialogue_output(self) -> None:
        agent = Agent(
            name="structured",
            instructions="parse",
            model="test",
            output_type=StructuredOutput,
        )

        with self.assertRaises(RuntimeError):
            _parse_output(agent, "下午两点到四点。")

    def test_structured_output_accepts_preface_and_trailing_json(self) -> None:
        agent = Agent(
            name="structured",
            instructions="parse",
            model="test",
            output_type=StructuredOutput,
        )

        output = _parse_output(
            agent,
            '_turn 是指响应轮次。\n最终JSON：{"answer": 7} {"answer": 8',
        )

        self.assertIsInstance(output, StructuredOutput)
        self.assertEqual(output.answer, 7)

    def test_structured_output_prefers_object_over_leading_list(self) -> None:
        agent = Agent(
            name="structured",
            instructions="parse",
            model="test",
            output_type=StructuredOutput,
        )

        output = _parse_output(
            agent,
            '命中失败标准：["未提及排名机制"]\n最终JSON：{"answer": 9}',
        )

        self.assertIsInstance(output, StructuredOutput)
        self.assertEqual(output.answer, 9)

    def test_runner_repairs_invalid_structured_output_once(self) -> None:
        agent = Agent(
            name="structured",
            instructions="parse",
            model="test",
            output_type=StructuredOutput,
        )

        with patch(
            "agents._chat_completion",
            side_effect=["[]", '{"answer": 11}'],
        ) as mock_chat:
            result = Runner.run_sync(agent, "input")

        self.assertEqual(result.final_output.answer, 11)
        self.assertEqual(mock_chat.call_count, 2)
        repair_messages = mock_chat.call_args_list[1].args[1]
        self.assertIn("上一次输出没有通过结构化校验", repair_messages[-1]["content"])


if __name__ == "__main__":
    unittest.main()
