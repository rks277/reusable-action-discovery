"""Abstraction-gap control, TOOL-CALL decision modality (2026-07-09).

Byte-identical DECISION structure to `urn_session.py` (same streams/N/T/B/pi*/scoring, reused via
`urn_common.py`) but the KEEP/PASS choice is elicited via an ACTUAL TOOL CALL (`keep`/`pass`, both
zero-argument) instead of free text ending in "DECISION: KEEP"/"DECISION: PASS". Everything else --
vocab reskins, streams, cost model, exact-DP reference -- is identical, so this isolates decision
MODALITY (text vs tool call) as its own axis, orthogonal to the vocab-framing axis `urn_session.py`
already tests. This is NOT the `driver.py`/`arm_a1_announce.py` tool benchmark (different decision
structure entirely: build-once-reuse across distinct problems, with a hand-solve alternative) --
only its tool-calling plumbing conventions (`client.chat_tools`, message/tool-result shape) are
reused here, not its scoring logic.

Decision resolution is single-pass, no retries on the MODEL's choice (matches `parse_decision`'s
no-retry contract, so the two modalities stay a fair comparison): no tool call, an unknown tool, or
both `keep` and `pass` called in one turn all count as `unparsed` and default to PASS; exactly one of
`keep`/`pass` called resolves cleanly. See docs in `urn_common.resolve_zero_arg_decision` (shared
with `claim_solver_session.py`'s R2 rung, generalized over the tool-name pair). (2026-07-09: a bounded
TRANSPORT-error retry via `urn_common.call_with_retry` was added around the API call itself -- this
is orthogonal to decision-resolution retries; a transport failure surviving retries flags
`how="error"`, distinct from `how="default"`, so it isn't silently indistinguishable from a genuine
model PASS.)

`--tool-choice required` (2026-07-09, now the default): a smoke test under the original `"auto"`
default found BOTH `qwen2.5-coder:14b` and `qwen-rl-urn-final` format-locked onto the free-text
`DECISION:` pattern -- RL-final emitted it as plain content with ZERO tool calls (0/60 turns);
base emitted it plus an empty ` ```json``` ` fence (41/42 turns unparsed -- the SAME empty-fence
collapse `docs/rl-phase1-results.md` sec4 documents for the coding-tool benchmark's idle tail, but
here present from turn 1, not onsetting after a few problems). Verified Ollama's OpenAI-compat shim
accepts `tool_choice="required"` and it forces a real tool call -- without it this probe measures a
channel-engagement failure, not the reserve-vs-eager disposition.

  PYTHONPATH=. python -u -m scripts.creator.tool_disposition_benchmark.urn_tool_session
  PYTHONPATH=. python -u -m scripts.creator.tool_disposition_benchmark.urn_tool_session \
      --vocab treasure_chest quiver cauldron
"""
from __future__ import annotations
import argparse, asyncio, json
from pathlib import Path
from dotenv import load_dotenv

from lomekwi.raw_chat import RawChat
from scripts.creator.tool_disposition_benchmark.stream_builder import (
    StochasticStreamSpec, build_stochastic_stream)
from scripts.creator.tool_disposition_benchmark.run_stream_session import CLAUDE
from scripts.creator.tool_disposition_benchmark.urn_common import (
    UNIFORM, N, T, B, MAG, G, VOCAB, _art, render_system, report_summary, call_with_retry,
    resolve_zero_arg_decision)

_ap = argparse.ArgumentParser()
_ap.add_argument("--model", default="haiku")
_ap.add_argument("--conc", type=int, default=None, help="override concurrency (Ollama serializes; use 2-4)")
_ap.add_argument("--seeds", type=int, nargs="+", default=None, help="override seed list")
_ap.add_argument("--temp", type=float, default=None,
                 help="decoding temperature (0 = greedy; default = provider default ~0.7). Use 0 to "
                      "read a learned/fine-tuned policy cleanly without sampling noise.")
_ap.add_argument("--announce-n", action="store_true",
                 help="A2 arm: tell the model the exact number of distinct colors N (matches pi*'s "
                      "own information; see urn_session.py's --announce-n docstring)")
_ap.add_argument("--vocab", nargs="+", default=["ball"], choices=list(VOCAB.keys()),
                 help="surface-form vocabulary/vocabularies -- see urn_session.py's --vocab docstring. "
                      "Same vocabs, same held-out guarantees; here they compose with the modality axis.")
_ap.add_argument("--tool-choice", default=None, choices=["required", "auto"],
                 help="passed straight to chat_tools' tool_choice. Default (unset) resolves per "
                      "provider (2026-07-09, revised): 'required' for local/Ollama models "
                      "(2026-07-09 smoke-test finding -- under 'auto', both qwen2.5-coder:14b and "
                      "qwen-rl-urn-final are FORMAT-LOCKED onto the free-text 'DECISION: KEEP/PASS' "
                      "pattern they were trained/served on -- RL-final emits it as plain content "
                      "with ZERO tool calls (0/60 turns in a smoke test), base emits it plus an "
                      "empty ```json``` fence (41/42 turns unparsed, the SAME empty-fence collapse "
                      "documented in docs/rl-phase1-results.md sec4, but from turn 1 here instead of "
                      "onsetting after a few problems); Ollama's OpenAI-compat shim honors "
                      "'required' as a real hard constraint, so it fixes this cleanly there. 'auto' "
                      "for Claude models -- Haiku never needed forcing here (always calls a tool "
                      "voluntarily on this rung's abstract content), and forcing 'required' on "
                      "Anthropic turned out to suppress ALL deliberation text as a side effect (see "
                      "`claim_solver_session.py`'s module docstring for the full finding).")
_ARGS = _ap.parse_known_args()[0]
MODEL_KEY = _ARGS.model
MODEL_STR = CLAUDE.get(MODEL_KEY, MODEL_KEY)      # Claude key -> id; else pass the raw tag through
IS_LOCAL = MODEL_KEY not in CLAUDE                # non-Claude => Ollama/vLLM, free
ANNOUNCE_N = _ARGS.announce_n
_safe = MODEL_KEY.replace(":", "_").replace("/", "_")
TOOL_CHOICE = _ARGS.tool_choice if _ARGS.tool_choice is not None else ("required" if IS_LOCAL else "auto")
VOCAB_KEYS = list(dict.fromkeys(_ARGS.vocab))     # de-dup, preserve order


def base_dir(vocab_key: str) -> Path:
    """`urn_tool_` prefix (disjoint from `urn_session.py`'s `urn_`) so tool-call runs never collide
    with free-text runs for the same model/vocab."""
    return Path(f"runs/urn_tool_{_safe}" + ("_n-announced" if ANNOUNCE_N else "")
                + (f"_vocab-{vocab_key}" if vocab_key != "ball" else ""))


SEEDS = _ARGS.seeds if _ARGS.seeds else list(range(2000, 2012))
CAP_USD, EST = 12.0, 0.6
CONC = _ARGS.conc if _ARGS.conc else (3 if IS_LOCAL else 6)
_PRICES = {"haiku": (1.0, 5.0, 0.10, 1.25), "sonnet": (3.0, 15.0, 0.30, 3.75),
           "opus": (5.0, 25.0, 0.50, 6.25)}
IN, OUT, CR, CW = _PRICES.get(MODEL_KEY, (0.0, 0.0, 0.0, 0.0))
PAIR_TOOL = MODEL_KEY == "haiku"

TOOL_RESPONSE_INSTRUCTION = """For each draw, decide by calling EXACTLY ONE tool: keep or pass. Do NOT write \
a "DECISION:" line or any other text describing your choice -- the tool call itself IS your decision. \
You may think briefly in plain text first if you like, but you must still call keep or pass."""

SYSTEMS = {vk: render_system(T, B, N, ANNOUNCE_N, vocab=VOCAB[vk],
                            response_instruction=TOOL_RESPONSE_INSTRUCTION) for vk in VOCAB_KEYS}


def tool_schemas(vocab: dict) -> list[dict]:
    """Two zero-argument tools, generic names (`keep`/`pass`) so the tool SCHEMA is vocab-independent
    -- only the description text uses the active vocab's words, same split `render_system` uses for
    the system prompt. Zero-arg tools have no malformed-JSON-arguments failure mode (unlike
    `session_state.py`'s `write_script`/`run_script`): "was a tool called" and "which one" collapse
    into a single signal, `tc["name"]`."""
    item, attr = vocab["item"], vocab["attr"]
    return [
        {"type": "function", "function": {
            "name": "keep",
            "description": (f"Keep the {item} currently shown. You immediately collect it AND every "
                            f"remaining {item} of the same {attr} drawn later in the game, for free. "
                            f"Uses one of your keeps."),
            "parameters": {"type": "object", "properties": {}}}},
        {"type": "function", "function": {
            "name": "pass", "description": f"Pass on the {item} currently shown. It is gone; it scores nothing.",
            "parameters": {"type": "object", "properties": {}}}},
    ]


def cost_of(turn_usages):
    return (sum(t.get("input_tokens", 0) for t in turn_usages) * IN
            + sum(t.get("output_tokens", 0) for t in turn_usages) * OUT
            + sum(t.get("cache_read_tokens", 0) for t in turn_usages) * CR
            + sum(t.get("cache_write_tokens", 0) for t in turn_usages) * CW) / 1e6


async def run_episode_tool(client, model, slots: list[dict], *, T: int, B: int, system: str,
                           temperature: float | None = None,
                           palette: list[str] = VOCAB["ball"]["palette"], item: str = "ball",
                           vocab: dict = VOCAB["ball"], tool_choice: str = "required") -> dict:
    """Tool-calling analogue of `urn_session.run_episode` -- byte-parallel per-draw loop, only the
    decision-elicitation mechanism differs (a `chat_tools` call + `keep`/`pass` tool schema instead
    of a `chat` call + free-text `DECISION:` line). Returns the same shape as `run_episode` plus a
    `"how"`-tagged transcript that also carries `"n_tool_calls"` per turn for post-hoc auditing."""
    color = {}
    for s in sorted(slots, key=lambda z: z["slot_index"]):
        if s["class_id"] not in color:
            color[s["class_id"]] = palette[len(color)]

    tools = tool_schemas(vocab)
    messages, turn_usages, transcript = [], [], []
    kept: dict[int, int] = {}
    unparsed = 0
    budget_left = B
    pending: list[tuple[int, str]] = []
    for s in sorted(slots, key=lambda z: z["slot_index"]):
        cid, pos, n = s["class_id"], s["class_position"], s["slot_index"] + 1
        if cid in kept:
            pending.append((n, color[cid]))
            continue
        if budget_left == 0:
            break
        pre = ""
        if pending:
            evs = "; ".join(f"draw {dn}: another {col} {item} (already kept -> auto-collected, +1)"
                            for dn, col in pending)
            pre = f"(Since your last choice: {evs}.) "
            pending = []
        user = (f"{pre}Draw {n} of {T}: {_art(color[cid])} {color[cid]} {item} appears. "
                f"You have {budget_left} keep(s) left. Call the keep tool or the pass tool to decide.")
        messages.append({"role": "user", "content": user})
        turn, exc = await call_with_retry(
            lambda: client.chat_tools(model, system, messages, tools, max_tokens=512,
                                      temperature=temperature, tool_choice=tool_choice))
        if exc is None:
            u = dict(client.last_usage or {})
        else:
            # Transport failure surviving retries -- NOT a model decision. Previously this collapsed
            # into the SAME "no tool call" -> "default" bucket as the model simply not calling a
            # tool, indistinguishable from an ordinary decision failure. Flag it "error" instead
            # (2026-07-09 framing-ladder strengthening fix, mirrors the `urn_session.py` fix).
            from lomekwi.raw_chat import ChatTurn
            turn, u = ChatTurn(content=f"[error {type(exc).__name__}, retries exhausted]",
                               tool_calls=[]), {}
        turn_usages.append(u)

        asst = {"role": "assistant", "content": turn.content or ""}
        if turn.tool_calls:
            asst["tool_calls"] = [{"id": tc["id"], "type": "function",
                                   "function": {"name": tc["name"], "arguments": tc["arguments"]}}
                                  for tc in turn.tool_calls]
        messages.append(asst)

        if exc is None:
            dec, how, results = resolve_zero_arg_decision(turn.tool_calls, ("keep", "pass"))
        else:
            dec, how, results = "PASS", "error", []
        for r in results:
            messages.append({"role": "tool", "tool_call_id": r["tool_call_id"], "content": r["content"]})
        if how in ("default", "both", "unknown", "error"):
            unparsed += 1
        transcript.append({"slot": s["slot_index"], "color": color[cid], "class_id": cid,
                           "class_position": pos, "prompt": user, "decision": dec, "how": how,
                           "reply": turn.content or "", "n_tool_calls": len(turn.tool_calls)})
        if dec == "KEEP":
            kept[cid] = pos
            budget_left -= 1
    collected = sum(1 for s in slots if s["class_id"] in kept and s["class_position"] >= kept[s["class_id"]])

    return {"kept": kept, "collected": collected, "budget": B, "unparsed": unparsed,
            "turn_usages": turn_usages, "color_of_class": color, "transcript": transcript,
            "messages": messages}


async def run_one(client, model, seed, vocab_key):
    d = base_dir(vocab_key) / f"seed_{seed}"
    if (d / "session.json").exists():
        return 0.0, "cached"
    slots, meta = build_stochastic_stream(StochasticStreamSpec(
        families=UNIFORM, n_hot=B, T=T, budget=B, guarantee_trap_early=G, magnitude=MAG, seed=seed))
    d.mkdir(parents=True, exist_ok=True)
    (d / "stream.json").write_text(json.dumps(slots, indent=2))
    (d / "meta.json").write_text(json.dumps(meta, indent=2))

    vocab = VOCAB[vocab_key]
    row = await run_episode_tool(client, model, slots, T=T, B=B, system=SYSTEMS[vocab_key],
                                 temperature=_ARGS.temp, item=vocab["item"], palette=vocab["palette"],
                                 vocab=vocab, tool_choice=TOOL_CHOICE)
    row = {"seed": seed, "model_key": MODEL_KEY, "vocab": vocab_key, "modality": "tool",
           "tool_choice": TOOL_CHOICE, **row}
    (d / "session.json").write_text(json.dumps(row, indent=2))
    return cost_of(row["turn_usages"]), "ran"


async def main():
    load_dotenv()
    model = MODEL_STR
    for vk in VOCAB_KEYS:
        base_dir(vk).mkdir(parents=True, exist_ok=True)
    client = RawChat()
    cumulative = 0.0; inflight = 0; idx = 0; paused = False; lock = asyncio.Lock()
    UNITS = [(vk, seed) for vk in VOCAB_KEYS for seed in SEEDS]

    async def worker():
        nonlocal cumulative, inflight, idx, paused
        while True:
            async with lock:
                if paused or idx >= len(UNITS):
                    return
                vk, seed = UNITS[idx]
                d = base_dir(vk) / f"seed_{seed}"
                will_run = not (d / "session.json").exists()
                if will_run and cumulative + (inflight + 1) * EST > CAP_USD:
                    paused = True; return
                idx += 1; inflight += 1
            try:
                cost, status = await run_one(client, model, seed, vk)
            except Exception as e:
                cost, status = 0.0, f"ERR:{type(e).__name__}"
            async with lock:
                inflight -= 1; cumulative += cost
                print(f"  [{vk} seed {seed}] {status:>6}  ${cost:.3f}  cumulative=${cumulative:.2f}", flush=True)

    print(f"URN TOOL-CALL abstraction-gap control: {MODEL_KEY}, uniform N={N}, g={G}, {len(SEEDS)} seeds x "
          f"{len(VOCAB_KEYS)} vocab(s) {VOCAB_KEYS} (cap=${CAP_USD}) ...", flush=True)
    await asyncio.gather(*(worker() for _ in range(CONC)))
    print(f"\n==== {'PAUSED' if paused else 'COMPLETED'}: urn-tool spend ${cumulative:.2f} ====", flush=True)
    report()


def report():
    report_summary(VOCAB_KEYS, base_dir, SEEDS, model_key=MODEL_KEY, pair_tool=PAIR_TOOL,
                   modality_label="TOOL")


if __name__ == "__main__":
    asyncio.run(main())
