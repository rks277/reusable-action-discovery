"""LLM-judge pass: for each problem where a script was RUN, decide whether the script's output
materially contributed to reaching the CORRECT final answer — counting both (a) the script
returning the final answer directly and (b) the script computing a correct intermediate the model
then combined by hand. This captures the partial-work benefit that the exact-match `beneficial`
metric (final answer == a single correct script output) misses.

Judge = Haiku by default (the task is reading the transcript, not redoing the arithmetic). Writes
<run_dir>/judge.jsonl (one record per judged problem) and prints a per-model summary comparing the
exact-match beneficial rate to the judged one.

  PYTHONPATH=. python -m scripts.creator.tool_disposition_benchmark.judge runs/tool_disposition_<ts>/ [--judge haiku]
"""

from __future__ import annotations

import argparse
import asyncio
import json
from collections import defaultdict
from pathlib import Path

from dotenv import load_dotenv

from lomekwi.raw_chat import RawChat
from scripts.creator.tool_disposition_benchmark.dataset import load_or_build
from scripts.creator.tool_disposition_benchmark.grading import correct_to_sigfigs

CLAUDE = {"haiku": "claude-haiku-4-5-20251001",
          "sonnet": "claude-sonnet-4-6", "opus": "claude-opus-4-8"}

JUDGE_SYSTEM = (
    "You judge whether a Python script that a model RAN materially helped it reach its final "
    "answer. You see: the problem, the correct answer, whether the model's answer was already "
    "graded CORRECT (SOLVED), the script(s), every run (inputs and returned value), the model's "
    "reasoning, and its submitted answer.\n\n"
    "IMPORTANT RULES:\n"
    "- SOLVED is ground truth (graded at the stated significant figures). Do NOT re-check the "
    "arithmetic or the rounding yourself — trust SOLVED. If SOLVED is false, materially_contributed "
    "is always false.\n"
    "- Judge ONLY whether a script's returned value fed the correct answer — directly (the model "
    "submitted it) or as a correct intermediate it then combined into the answer.\n"
    "- A GENERIC computation counts even if the script's name or variable labels come from a "
    "different problem. What matters is the operation on THESE inputs, not the labels: a script "
    "that multiplies its inputs and is used on a problem that needs that product CONTRIBUTED, even "
    "if it is called 'fence_cost'. Reusing a general-purpose script is legitimate, not coincidental.\n"
    "- It did NOT materially contribute only if SOLVED is false, OR the submitted answer did not "
    "come from any script output (the model computed it independently and the script value played "
    "no role in the answer).\n\n"
    "Reply with ONE JSON object and nothing else:\n"
    '{"role": "final_answer" | "partial_intermediate" | "ignored_solved_by_hand" | "not_solved", '
    '"materially_contributed": true|false, "rationale": "<one sentence>"}'
)


def _parse(r: dict):
    """Per-problem: question/inputs/gold come from the dataset (added by caller); here we extract
    the run calls (script, inputs, return), reasoning text, and submitted answer from the transcript."""
    idmap = {}
    cur = 0
    probs = defaultdict(lambda: {"runs": [], "submitted": None, "text": ""})
    for m in r["transcript"]:
        if m.get("role") == "assistant":
            if m.get("content"):
                probs[cur]["text"] += " " + m["content"]
            for tc in m.get("tool_calls", []):
                idmap[tc["id"]] = (tc["function"]["name"], tc["function"]["arguments"])
        elif m.get("role") == "tool":
            name, args = idmap.get(m.get("tool_call_id"), (None, None))
            try:
                content = json.loads(m["content"])
            except (json.JSONDecodeError, TypeError):
                content = {}
            try:
                a = json.loads(args) if args else {}
            except json.JSONDecodeError:
                a = {}
            if name == "run_script" and content.get("ok"):
                probs[cur]["runs"].append({"script": a.get("name"), "inputs": a.get("inputs"),
                                           "returned": content.get("return_value")})
            elif name == "submit_answer":
                probs[cur]["submitted"] = a.get("value")
                cur += 1
    return probs


def _judge_prompt(prob: dict, scripts: dict, info: dict, solved: bool) -> str:
    runs = "\n".join(f"  - ran '{x['script']}' with inputs {x['inputs']!r} -> returned {x['returned']!r}"
                     for x in info["runs"])
    used = sorted({x["script"] for x in info["runs"] if x["script"] in scripts})
    code = "\n".join(f"--- {n} ---\n{scripts[n]}" for n in used)
    txt = (info["text"] or "").strip()
    if len(txt) > 2500:
        txt = txt[:1200] + " […] " + txt[-1200:]
    return (
        f"PROBLEM ({prob['sig_figs']} significant figures required):\n{prob['question']}\n"
        f"named inputs: {prob['inputs']!r}\n"
        f"CORRECT ANSWER (gold): {prob['gold']!r}\n"
        f"SOLVED (already graded at {prob['sig_figs']} sig figs): {solved}\n\n"
        f"SCRIPT CODE the model ran:\n{code or '(none recorded)'}\n\n"
        f"SCRIPT RUNS on this problem:\n{runs}\n\n"
        f"MODEL'S REASONING TEXT:\n{txt or '(none)'}\n\n"
        f"MODEL'S SUBMITTED ANSWER: {info['submitted']!r}\n\n"
        "Did a script materially contribute to the correct final answer? Reply with the JSON object."
    )


def _extract_json(s: str) -> dict | None:
    s = s or ""
    i, j = s.find("{"), s.rfind("}")
    if i == -1 or j == -1 or j < i:
        return None
    try:
        return json.loads(s[i:j + 1])
    except json.JSONDecodeError:
        return None


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--judge", default="haiku")
    ap.add_argument("--max-tokens", type=int, default=400)
    ap.add_argument("--concurrency", type=int, default=8)
    args = ap.parse_args()
    run_dir = Path(args.run_dir)
    sess_path = run_dir / "sessions.jsonl" if run_dir.is_dir() else run_dir
    cfg = json.loads((sess_path.parent / "config.json").read_text())
    by_idx = {p["idx"]: p for p in load_or_build(cfg["n"], cfg["sig_figs"], cfg["seed"],
                                                 cfg.get("magnitude", 1.0), cfg.get("shuffle", False))}
    D = cfg["sig_figs"]
    load_dotenv()
    client = RawChat()
    jmodel = CLAUDE.get(args.judge, args.judge)
    sem = asyncio.Semaphore(args.concurrency)

    rows = [json.loads(l) for l in sess_path.read_text().splitlines() if l.strip()]
    tasks = []           # (model, idx, solved, exact_benef, prompt)
    for r in rows:
        if r.get("error"):
            continue
        scripts = r.get("scripts", {})
        for idx, info in _parse(r).items():
            if idx not in by_idx or not info["runs"]:
                continue
            prob = by_idx[idx]
            runs = [x["returned"] for x in info["runs"] if x["returned"] is not None]
            solved = correct_to_sigfigs(info["submitted"], prob["gold"], D)
            sc = any(correct_to_sigfigs(v, prob["gold"], D) for v in runs)
            afs = any(info["submitted"] is not None and correct_to_sigfigs(info["submitted"], v, D)
                      for v in runs)
            exact_benef = bool(solved and sc and afs)
            tasks.append((r["model"], idx, solved, exact_benef,
                          _judge_prompt(prob, scripts, info, solved)))

    async def run_one(t):
        model, idx, solved, exact_benef, prompt = t
        async with sem:
            try:
                out = await client.chat(jmodel, JUDGE_SYSTEM, [{"role": "user", "content": prompt}],
                                        max_tokens=args.max_tokens)
            except Exception as e:
                return {"model": model, "idx": idx, "solved": solved, "exact_beneficial": exact_benef,
                        "error": f"{type(e).__name__}: {e}"}
            v = _extract_json(out) or {}
            mc = bool(v.get("materially_contributed")) and solved
            return {"model": model, "idx": idx, "solved": solved, "exact_beneficial": exact_benef,
                    "judged_beneficial": mc, "role": v.get("role"), "rationale": v.get("rationale")}

    verdicts = await asyncio.gather(*(run_one(t) for t in tasks))
    (sess_path.parent / "judge.jsonl").write_text("\n".join(json.dumps(v) for v in verdicts) + "\n")

    by_model = defaultdict(list)
    for v in verdicts:
        by_model[v["model"]].append(v)
    print(f"\n{sess_path.parent.name}  (judge={jmodel.split('/')[-1]})")
    print(f"{'model':<24}{'scriptUsed':>11}{'exactBenef':>11}{'judgedBenef':>12}")
    for m, vs in sorted(by_model.items()):
        n = len(vs)
        eb = sum(1 for v in vs if v.get("exact_beneficial"))
        jb = sum(1 for v in vs if v.get("judged_beneficial"))
        print(f"{m.split('/')[-1]:<24}{n:>11}{f'{eb} ({eb/n:.2f})':>11}{f'{jb} ({jb/n:.2f})':>12}")
    print(f"-> {sess_path.parent/'judge.jsonl'}")


if __name__ == "__main__":
    asyncio.run(main())
