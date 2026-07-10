"""Cross-rung parity self-test for the framing ladder (`docs/framing-ladder-spec.md` §5's
falsification criterion: "R2 must be byte-structurally parallel to R0/R1 apart from problem
rendering and tool names -- verify this claim with an executable parity test").

Feeds R0 (`urn_session.run_episode`, free-text), R1 (`urn_tool_session.run_episode_tool`,
`keep`/`pass` tool calls), R2 (`claim_solver_session.run_episode_claim`, `claim_solver`/`skip_solver`
tool calls), and R2c (`claim_solver_code_session.run_episode_code_claim`, same tool pair but
`claim_solver` now requires a `code` argument) the IDENTICAL hand-built stream and an equivalent
scripted decision sequence (commit/decline pattern held fixed across all four modalities; only the
surface spelling of the decision -- and, for R2c, the presence of a code argument on commit -- differs
per rung), then asserts the four loops produce identical commitment positions, collected/auto-solved
totals, budgets, and unresolved-decision counts. This is the executable check the original plan only
asserted in prose ("byte-structurally parallel") -- turning an eyeballed claim into a falsifiable one.

No network calls. Run with:
  PYTHONPATH=. python -m scripts.creator.tool_disposition_benchmark.ladder_parity_selftest
"""
from __future__ import annotations
import asyncio, json

from lomekwi.raw_chat import ChatTurn
from scripts.creator.tool_disposition_benchmark.urn_common import VOCAB
from scripts.creator.tool_disposition_benchmark.urn_session import run_episode
from scripts.creator.tool_disposition_benchmark.urn_tool_session import run_episode_tool
from scripts.creator.tool_disposition_benchmark.claim_solver_session import (
    run_episode_claim, render_system_claim)
from scripts.creator.tool_disposition_benchmark.claim_solver_code_session import (
    run_episode_code_claim, render_system_code_claim)

# Same 9-slot class sequence and commit/decline pattern as `claim_solver_session._selftest` --
# exercises skip, first-sight claim, delayed claim, auto-solve notices, and budget exhaustion
# simultaneously, across all three rungs at once.
CLASS_SEQ = [0, 0, 1, 2, 0, 2, 3, 3, 3]
B = 2
COMMIT_PATTERN = [True, False, False, True]   # turn i: commit (KEEP/CLAIM_SOLVER) vs decline


def _slots():
    return [{"slot_index": i, "class_id": cid, "class_position": CLASS_SEQ[:i + 1].count(cid),
            "question": f"synthetic problem for class {cid} (slot {i})"}
            for i, cid in enumerate(CLASS_SEQ)]


class ScriptedTextClient:
    """R0: free-text `DECISION: KEEP`/`DECISION: PASS`, driven by the same commit/decline pattern."""
    def __init__(self, pattern):
        self.plan = ["DECISION: KEEP" if c else "DECISION: PASS" for c in pattern]
        self.last_usage = {"input_tokens": 1, "output_tokens": 1}

    async def chat(self, model, system, messages, max_tokens, temperature):
        return self.plan.pop(0)


class ScriptedToolClient:
    """R1/R2: tool-call decision, driven by the same commit/decline pattern with a caller-supplied
    (commit_name, decline_name) pair -- (`keep`, `pass`) for R1, (`claim_solver`, `skip_solver`) for
    R2."""
    def __init__(self, pattern, names: tuple[str, str]):
        commit_name, decline_name = names
        self.plan = [commit_name if c else decline_name for c in pattern]
        self.calls = 0
        self.last_usage = {"input_tokens": 1, "output_tokens": 1}

    async def chat_tools(self, model, system, messages, tools, max_tokens, temperature, tool_choice):
        self.calls += 1
        name = self.plan.pop(0)
        return ChatTurn(content="", tool_calls=[{"id": f"c{self.calls}", "name": name, "arguments": "{}"}])


class ScriptedCodeToolClient:
    """R2c: same commit/decline pattern, but every commit call attaches a (trivial, distinct) `code`
    argument -- content doesn't matter for this parity test (correctness is graded separately, never
    scored -- see `claim_solver_code_session` module docstring), only that the mechanics accept it and
    still resolve to the same commit/decline outcome as R1/R2's zero-argument calls."""
    def __init__(self, pattern, names: tuple[str, str]):
        commit_name, decline_name = names
        self.plan = [commit_name if c else decline_name for c in pattern]
        self.calls = 0
        self.last_usage = {"input_tokens": 1, "output_tokens": 1}

    async def chat_tools(self, model, system, messages, tools, max_tokens, temperature, tool_choice):
        self.calls += 1
        name = self.plan.pop(0)
        args = {"code": f"def solve(inputs): return {self.calls}"} if name == "claim_solver" else {}
        return ChatTurn(content="", tool_calls=[{"id": f"c{self.calls}", "name": name,
                                                 "arguments": json.dumps(args), "args": args}])


def _selftest():
    async def main():
        slots = _slots()
        T, vocab = len(slots), VOCAB["ball"]

        row0 = await run_episode(ScriptedTextClient(COMMIT_PATTERN), "fake", slots, T=T, B=B,
                                 system="irrelevant for a scripted client", temperature=0.0,
                                 palette=vocab["palette"], item=vocab["item"])
        row1 = await run_episode_tool(ScriptedToolClient(COMMIT_PATTERN, ("keep", "pass")), "fake",
                                      slots, T=T, B=B, system="irrelevant", temperature=0.0,
                                      palette=vocab["palette"], item=vocab["item"], vocab=vocab,
                                      tool_choice="required")
        row2 = await run_episode_claim(ScriptedToolClient(COMMIT_PATTERN, ("claim_solver", "skip_solver")),
                                       "fake", slots, T=T, B=B,
                                       system=render_system_claim(T, B, 4, True), temperature=0.0,
                                       tool_choice="required")
        row2c = await run_episode_code_claim(
            ScriptedCodeToolClient(COMMIT_PATTERN, ("claim_solver", "skip_solver")), "fake", slots,
            T=T, B=B, system=render_system_code_claim(T, B, 4, True), temperature=0.0,
            tool_choice="required")

        # 1. identical commitment positions (R0/R1's `kept`, R2/R2c's `claimed` -- same shape).
        assert row0["kept"] == row1["kept"] == row2["claimed"] == row2c["claimed"] == {0: 1, 2: 2}, \
            (row0["kept"], row1["kept"], row2["claimed"], row2c["claimed"])

        # 2. identical auto-collection/auto-solve + terminal totals.
        assert row0["collected"] == row1["collected"] == row2["collected"] == row2c["collected"] == 4, \
            (row0["collected"], row1["collected"], row2["collected"], row2c["collected"])

        # 3. identical budgets consumed.
        assert row0["budget"] == row1["budget"] == row2["budget"] == row2c["budget"] == B

        # 4. identical unresolved-decision counts (all clean here -- error/default/malformed_args
        #    paths are covered separately by each module's own self-test, not re-tested here).
        assert row0["unparsed"] == row1["unparsed"] == row2["unparsed"] == row2c["unparsed"] == 0

        # 5. identical NUMBER of decision turns actually generated (budget exhaustion truncates the
        #    stream identically in all four -- slots 6/7/8 never presented in any rung).
        n0, n1, n2, n2c = (len(row0["transcript"]), len(row1["transcript"]), len(row2["transcript"]),
                          len(row2c["transcript"]))
        assert n0 == n1 == n2 == n2c == len(COMMIT_PATTERN), (n0, n1, n2, n2c)

        # 6. identical commit/decline pattern, turn-by-turn, independent of surface spelling.
        commit0 = [t["decision"] == "KEEP" for t in row0["transcript"]]
        commit1 = [t["decision"] == "KEEP" for t in row1["transcript"]]
        commit2 = [t["decision"] == "CLAIM_SOLVER" for t in row2["transcript"]]
        commit2c = [t["decision"] == "CLAIM_SOLVER" for t in row2c["transcript"]]
        assert commit0 == commit1 == commit2 == commit2c == COMMIT_PATTERN, \
            (commit0, commit1, commit2, commit2c)

        # 7. R2c-specific: the code argument was actually captured against the right class, and
        #    doesn't leak into the other rungs' commitment dict shape.
        assert set(row2c["claimed_code"]) == {0, 2}, row2c["claimed_code"]

        print("ladder_parity_selftest OK -- R0/R1/R2/R2c produce IDENTICAL commitment positions "
              f"({row0['kept']}), collected totals ({row0['collected']}), budgets ({row0['budget']}), "
              "and commit/decline patterns, on the identical stream, under an equivalent scripted "
              "decision sequence. R2 and R2c are both byte-structurally parallel to R0/R1 on every "
              "mechanic this test can exercise (skip / first-sight claim / delayed claim / "
              "auto-solve / budget exhaustion) -- only problem rendering, tool/action names, and "
              "(for R2c only) the required code argument differ, as required.")

    asyncio.run(main())


def _selftest_economic_surface_parity():
    """Extends the above to the economic response surface (`docs/economic-response-surface-spec.md`):
    R0 and R2c, fed the SAME scripted commit/decline pattern and stream, at several (B,K) combinations,
    must produce identical commitment positions, budgets, and NET SCORES (`economic_surface.net_score`)
    -- charge only changes prompt text (`render_system`/`render_system_code_claim`'s `charge` param),
    never the mechanics, so parity here is a check on `economic_surface.py`'s scoring layer, not on
    the run_episode* loops (already covered above)."""
    from scripts.creator.tool_disposition_benchmark.economic_surface import net_score

    async def main():
        for B, K in [(2, 0), (2, 5), (1, 24)]:
            pattern = COMMIT_PATTERN if B == 2 else [True, False, False, False, False, False, False, False, False]
            slots = _slots()
            row0 = await run_episode(ScriptedTextClient(pattern), "fake", slots, T=len(slots), B=B,
                                     system="irrelevant", temperature=0.0,
                                     palette=VOCAB["ball"]["palette"], item=VOCAB["ball"]["item"])
            row2c = await run_episode_code_claim(
                ScriptedCodeToolClient(pattern, ("claim_solver", "skip_solver")), "fake", slots,
                T=len(slots), B=B, system=render_system_code_claim(len(slots), B, 4, True, charge=K),
                temperature=0.0, tool_choice="required")

            assert row0["kept"] == row2c["claimed"], (B, K, row0["kept"], row2c["claimed"])
            assert row0["collected"] == row2c["collected"], (B, K, row0["collected"], row2c["collected"])
            assert row0["budget"] == row2c["budget"] == B

            s0, s2c = net_score(row0, K), net_score(row2c, K)
            assert s0 == s2c, (B, K, s0, s2c)
            # sanity: net score matches the raw formula directly, not just "R0 == R2c".
            assert s0 == row0["collected"] - K * len(row0["kept"]), (B, K, s0, row0["collected"])
        print("ladder_parity_selftest (economic surface) OK -- R0/R2c net scores match exactly "
              "across (B,K) in {(2,0), (2,5), (1,24)}: charge only changes prompt text, never "
              "commitment positions, budgets, or the resulting net score.")

    asyncio.run(main())


if __name__ == "__main__":
    _selftest()
    _selftest_economic_surface_parity()
