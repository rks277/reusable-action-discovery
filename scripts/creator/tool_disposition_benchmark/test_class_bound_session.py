from __future__ import annotations

import json
import unittest

from scripts.creator.tool_disposition_benchmark.driver import run_session
from scripts.creator.tool_disposition_benchmark.prompts import (
    CLASS_BOUND_SCRIPT_NOTE, problem_prompt, system_prompt)
from scripts.creator.tool_disposition_benchmark.session_state import (
    CLASS_BOUND_VERSION, SessionState, TOOL_SCHEMAS)


CODE = "def solve(inputs: dict) -> float:\n    return inputs['x'] * 2"


def problem(idx: int, class_id: int, x: int = 2) -> dict:
    return {
        "idx": idx,
        "item_idx": idx,
        "keys": ["x"],
        "question": f"Double {x}.",
        "inputs": {"x": x},
        "gold": x * 2,
        "sig_figs": 1,
        "exact_int": True,
        "_class_id": class_id,
        "_family": f"family_{class_id}",
    }


class ClassBoundStateTest(unittest.TestCase):
    def test_prompt_and_schema_disclose_scope_without_hidden_labels(self):
        p = problem(0, 73)
        rendered = problem_prompt(p, 1, 1)
        self.assertNotIn("73", rendered)
        self.assertNotIn("_class_id", rendered)
        system = system_prompt(1, 1) + CLASS_BOUND_SCRIPT_NOTE
        self.assertIn("bound to the hidden problem type", system)
        schema = json.dumps(TOOL_SCHEMAS(class_bound=True))
        self.assertIn("same hidden type", schema)

    def test_same_class_runs_and_cross_class_is_refused(self):
        state = SessionState(
            problems=[problem(0, 4, 2), problem(1, 4, 3), problem(2, 9, 5)],
            budget=2,
        )
        self.assertTrue(state.class_bound)
        self.assertTrue(state.op_write_script("double", CODE)["ok"])
        self.assertEqual(state.op_run_script("double", {"x": 2})["return_value"], 4)
        state.op_submit_answer(4)
        self.assertEqual(state.op_run_script("double", {"x": 3})["return_value"], 6)
        state.op_submit_answer(6)
        refused = state.op_run_script("double", {"x": 5})
        self.assertFalse(refused["ok"])
        self.assertIn("different hidden problem type", refused["error"])
        self.assertNotIn("4", refused["error"])
        self.assertNotIn("9", refused["error"])
        score = state.score()
        self.assertEqual(score["benchmark_version"], CLASS_BOUND_VERSION)
        self.assertEqual(score["n_cross_class_run_refused"], 1)
        self.assertFalse(score["records"][2]["used_script"])
        self.assertEqual(score["persistence"], 2.0)

    def test_replacement_consumes_budget_and_rebinds_artifact(self):
        state = SessionState(
            problems=[problem(0, 1), problem(1, 2)],
            budget=2,
        )
        state.op_write_script("solver", CODE)
        first_artifact = state.script_artifact["solver"]
        state.op_submit_answer(4)
        state.op_write_script("solver", CODE)
        second_artifact = state.script_artifact["solver"]
        self.assertNotEqual(first_artifact, second_artifact)
        self.assertEqual(state.n_write_calls, 2)
        self.assertEqual(state.script_class["solver"], 2)
        self.assertEqual(len(state.script_writes), 2)
        self.assertTrue(state.script_writes[1]["replaced"])

    def test_global_sessions_remain_backward_compatible(self):
        problems = [
            {k: v for k, v in problem(0, 1).items() if not k.startswith("_")},
            {k: v for k, v in problem(1, 2).items() if not k.startswith("_")},
        ]
        state = SessionState(problems=problems, budget=1)
        self.assertFalse(state.class_bound)
        state.op_write_script("double", CODE)
        state.op_submit_answer(4)
        self.assertTrue(state.op_run_script("double", {"x": 2})["ok"])


class _Turn:
    def __init__(self, tool_calls):
        self.content = ""
        self.tool_calls = tool_calls
        self.finish_reason = "tool_calls"


class _FakeClient:
    def __init__(self):
        self.last_usage = None
        self.last_response_model = "fake"
        self.system = None
        self.tools = None
        self.turn = 0

    async def chat_tools(self, model, system, messages, tools, **_kwargs):
        self.system = system
        self.tools = tools
        self.turn += 1
        self.last_usage = {"input_tokens": 10, "output_tokens": 10}
        calls = [
            {
                "id": f"w{self.turn}",
                "name": "write_script",
                "arguments": json.dumps({"name": f"s{self.turn}", "code": CODE}),
                "args": {"name": f"s{self.turn}", "code": CODE},
            },
            {
                "id": f"a{self.turn}",
                "name": "submit_answer",
                "arguments": json.dumps({"value": 4}),
                "args": {"value": 4},
            },
        ]
        return _Turn(calls)


class ClassBoundDriverTest(unittest.IsolatedAsyncioTestCase):
    async def test_driver_discloses_scope_and_stops_after_three_writes(self):
        state = SessionState(
            problems=[problem(i, i) for i in range(4)],
            budget=3,
        )
        client = _FakeClient()
        row = await run_session(
            client,
            "fake",
            state,
            token_cap=10_000,
            stop_on_budget_exhausted=True,
        )
        self.assertTrue(row["stopped_on_budget"])
        self.assertEqual(row["n_scripts_written"], 3)
        self.assertEqual(row["n_turns"], 3)
        self.assertIn(CLASS_BOUND_SCRIPT_NOTE.strip(), client.system)
        self.assertIn("same hidden type", json.dumps(client.tools))


if __name__ == "__main__":
    unittest.main()
