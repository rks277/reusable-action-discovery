"""Throwaway diagnostic: does the FT model use tools under the EXACT harness prompt? Compare
eval-format (tools= passed -> Qwen injects a <tools> block, as the harness does) vs training-format
(no tools= -> no block, as train_lora rendered the demos). Run on the box against localhost:8000."""
from openai import OpenAI

from scripts.creator.tool_disposition_benchmark.prompts import (
    RECURRENCE_NOTE, n_types_note, system_prompt)
from scripts.creator.tool_disposition_benchmark.session_state import TOOL_SCHEMAS

client = OpenAI(base_url="http://localhost:8000/v1", api_key="EMPTY")
SYSTEM = system_prompt(60, 3, None) + RECURRENCE_NOTE + n_types_note(8)
USER = ("PROBLEM 1 of 60: A register starts at X = 87. It is updated 6 times; each update replaces X "
        "with ((2200 * X) + 2400) mod 46400. Report the final X.")
msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": USER}]
tools = TOOL_SCHEMAS()

print("=== TEST A: eval-format (tools= passed, as the harness does) ===")
r = client.chat.completions.create(model="qwen-ft-pistar", messages=msgs, tools=tools,
                                   tool_choice="auto", max_tokens=4096, temperature=0.6)
ch = r.choices[0]
print("finish_reason:", ch.finish_reason, "| completion_tokens:", r.usage.completion_tokens)
print("content[:250]:", repr((ch.message.content or "")[:250]))
tc = ch.message.tool_calls
print("tool_calls:", [(t.function.name, t.function.arguments[:100]) for t in tc] if tc else None)

print("\n=== TEST B: training-format (NO tools=, look for <tool_call> in text) ===")
r2 = client.chat.completions.create(model="qwen-ft-pistar", messages=msgs,
                                    max_tokens=4096, temperature=0.6)
ch2 = r2.choices[0]
c2 = ch2.message.content or ""
print("finish_reason:", ch2.finish_reason, "| completion_tokens:", r2.usage.completion_tokens)
print("has <tool_call>:", "<tool_call>" in c2)
print("content[:400]:", repr(c2[:400]))
