"""Abstraction-gap control (the "urn / balls" isomorph of the tool task).

Byte-identical DECISION to the tool benchmark, stripped of the tool cover story: balls of N colors are
drawn one at a time (the SAME stochastic stream as the tool runs, paired by seed); at <=B draws the
model may KEEP the current ball, which collects it plus every remaining ball of that color for free
(= build-a-tool-then-reuse). Passed balls, and colors never kept, score nothing (= hand-solve at
a_hand=0). Reference pi* is literally the same exact-DP; keeps map onto `model_builds`.

Headline metric = KEEP LATENESS (does it keep a color on FIRST sighting = eager, or reserve for a
color that has recurred?) compared to the tool task's build lateness. If the model RESERVES here but
builds eagerly in the tool task -> the tool/coding framing suppresses recognition (recognition failure,
not competence). If it is eager here too -> domain-general greedy myopia.

Clear color labels (approved). A0 information: told T and B, NOT which colors are frequent nor N.

--vocab (2026-07-09, extended 2026-07-09) swaps the surface wording for a framing-generalization probe:
same N/T/B/streams/pi*, only the words differ. Three HELD-OUT vocabs the RL run never trained on (RL
used only "ball"; SFT's 7-vocab rotation never used ANY of these three vessel/item/attr combos, not just
a different pairing of them): 'treasure_chest' (coins in a chest, attr="crest"), 'quiver' (arrows in a
quiver, attr="feather"), 'cauldron' (potions in a cauldron, attr="label"). Pass multiple --vocab values
to run all three and get BOTH a per-vocab breakdown AND the pooled average across them in `report()` --
averaging answers "does the disposition generalize across re-skins", not just one arbitrarily-chosen one.

  PYTHONPATH=. python -u -m scripts.creator.tool_disposition_benchmark.urn_session
  PYTHONPATH=. python -u -m scripts.creator.tool_disposition_benchmark.urn_session \
      --vocab treasure_chest quiver cauldron
"""
from __future__ import annotations
import argparse, asyncio, json, re
from pathlib import Path
from dotenv import load_dotenv

from lomekwi.raw_chat import RawChat
from scripts.creator.tool_disposition_benchmark.stream_builder import (
    StochasticStreamSpec, build_stochastic_stream)
from scripts.creator.tool_disposition_benchmark.run_stream_session import CLAUDE
from scripts.creator.tool_disposition_benchmark.urn_common import (
    UNIFORM, N, T, B, MAG, G, VOCAB, _art, render_system, _balls_collected,
    make_costs_and_dp, report_summary, call_with_retry)

_ap = argparse.ArgumentParser()
# --model accepts a Claude key (haiku/sonnet/opus) OR any raw model tag. A tag containing ":"
# (e.g. "qwen2.5-coder:7b") routes to Ollama via RawChat (set OLLAMA_BASE_URL); local models are
# free, so pricing/spend-guard are disabled for them.
_ap.add_argument("--model", default="haiku")
_ap.add_argument("--conc", type=int, default=None, help="override concurrency (Ollama serializes; use 2-4)")
_ap.add_argument("--seeds", type=int, nargs="+", default=None, help="override seed list")
_ap.add_argument("--temp", type=float, default=None,
                 help="decoding temperature (0 = greedy; default = provider default ~0.7). Use 0 to "
                      "read a learned/fine-tuned policy cleanly without sampling noise.")
_ap.add_argument("--announce-n", action="store_true",
                 help="A2 arm: tell the model the exact number of distinct colors N (matches pi*'s "
                      "own information -- pi*'s Dirichlet-multinomial predictive (alpha+k)/(N*alpha+t) "
                      "is CONSTRUCTED WITH exact N, so without this flag the 'same-information' "
                      "regret comparison is not actually same-information; see 2026-07-03 audit)")
_ap.add_argument("--vocab", nargs="+", default=["ball"], choices=list(VOCAB.keys()),
                 help="surface-form vocabulary/vocabularies for the game (2026-07-09 framing-"
                      "generalization probe, extended same day to multiple vocabs): 'ball' is the exact "
                      "wording RL/SFT were trained/evaluated on. 'treasure_chest'/'quiver'/'cauldron' "
                      "are HELD-OUT surface forms the RL run never saw (RL trained only on 'ball'; SFT's "
                      "7-vocab rotation never used any of these three combos). Passing more than one "
                      "runs seeds x vocabs and report() prints both a per-vocab breakdown AND the pooled "
                      "average across them -- tests whether the reserve disposition is bound to the "
                      "literal ball/bag/color wording or generalizes across a same-structure re-skin, "
                      "and whether that generalization is itself vocab-specific or robust. Same "
                      "N/T/B/streams/pi* either way -- decision structure is untouched, only the words are.")
_ARGS = _ap.parse_known_args()[0]
MODEL_KEY = _ARGS.model
MODEL_STR = CLAUDE.get(MODEL_KEY, MODEL_KEY)      # Claude key -> id; else pass the raw tag through
IS_LOCAL = MODEL_KEY not in CLAUDE                # non-Claude => Ollama/vLLM, free
ANNOUNCE_N = _ARGS.announce_n
_safe = MODEL_KEY.replace(":", "_").replace("/", "_")
VOCAB_KEYS = list(dict.fromkeys(_ARGS.vocab))     # de-dup, preserve order


def base_dir(vocab_key: str) -> Path:
    return Path(f"runs/urn_{_safe}" + ("_n-announced" if ANNOUNCE_N else "")
                + (f"_vocab-{vocab_key}" if vocab_key != "ball" else ""))


SEEDS = _ARGS.seeds if _ARGS.seeds else list(range(2000, 2012))  # 12 -- same streams as Haiku/Opus urn
CAP_USD, EST = 12.0, 0.6
CONC = _ARGS.conc if _ARGS.conc else (3 if IS_LOCAL else 6)      # Ollama loads one model; keep low
_PRICES = {"haiku": (1.0, 5.0, 0.10, 1.25), "sonnet": (3.0, 15.0, 0.30, 3.75),
           "opus": (5.0, 25.0, 0.50, 6.25)}       # $/1e6 (in, out, cache_read, cache_write)
IN, OUT, CR, CW = _PRICES.get(MODEL_KEY, (0.0, 0.0, 0.0, 0.0))   # local models: no cost
PAIR_TOOL = MODEL_KEY == "haiku"          # the A1 tool baseline (runs/arm_a1_announce) is Haiku-only

SYSTEMS = {vk: render_system(T, B, N, ANNOUNCE_N, vocab=VOCAB[vk]) for vk in VOCAB_KEYS}


def cost_of(turn_usages):
    return (sum(t.get("input_tokens", 0) for t in turn_usages) * IN
            + sum(t.get("output_tokens", 0) for t in turn_usages) * OUT
            + sum(t.get("cache_read_tokens", 0) for t in turn_usages) * CR
            + sum(t.get("cache_write_tokens", 0) for t in turn_usages) * CW) / 1e6


def parse_decision(text: str):
    """Return (decision, how). Robust to small models that emit a bare 'KEEP'/'PASS' + rambling
    (which often contains the other word, e.g. '...nothing to pass') instead of the DECISION: line.
    Priority: (1) explicit DECISION: line (last wins); (2) a LEADING KEEP/PASS token; (3) first
    standalone token anywhere; (4) default PASS. `how` == 'default' flags a true parse failure."""
    t = (text or "").strip()
    hits = re.findall(r"DECISION:\s*(KEEP|PASS)", t, re.I)
    if hits:
        return hits[-1].upper(), "tag"
    m = re.match(r"[\W]*(KEEP|PASS)\b", t, re.I)
    if m:
        return m.group(1).upper(), "lead"
    m2 = re.search(r"\b(KEEP|PASS)\b", t, re.I)
    if m2:
        return m2.group(1).upper(), "scan"
    return "PASS", "default"


async def run_episode(client, model, slots: list[dict], *, T: int, B: int, system: str,
                      temperature: float | None = None, palette: list[str] = VOCAB["ball"]["palette"],
                      item: str = "ball", reasoning_effort: str | None = None,
                      stop_after_turn=None) -> dict:
    """Run ONE urn episode against a pre-built `slots` stream -- no file I/O, no caching, so callers
    (the CLI eval below, and `rl_rollout.py`'s RL rollout collector) own persistence. Extracted from
    `run_one` (2026-07-07, RL Phase 1) so both reuse the EXACT same per-draw loop byte-for-byte, rather
    than a paraphrased reimplementation (this project's standing lesson: a near-miss reimplementation is
    as bad as no fix at all). Returns `{kept, collected, unparsed, turn_usages, color_of_class,
    transcript, messages}` -- `messages` (OpenAI-format user/assistant turns) is new relative to the
    original inline version, needed for RL's `build_example`-based log-prob masking. `item` (2026-07-09)
    only changes the per-draw wording ("ball"/"coin"/...) to match `system`'s vocab -- callers that omit
    it (RL/SFT) get the original "ball" text unchanged."""
    # clear color labels assigned by FIRST-APPEARANCE order (leaks nothing about rate/role)
    color = {}
    for s in sorted(slots, key=lambda z: z["slot_index"]):
        if s["class_id"] not in color:
            color[s["class_id"]] = palette[len(color)]

    messages, turn_usages, transcript = [], [], []
    kept: dict[int, int] = {}                  # class_id -> class_position at which kept
    unparsed = 0
    budget_left = B
    safety_stop = None
    pending: list[tuple[int, str]] = []        # kept-color draws seen since the last decision prompt
    for s in sorted(slots, key=lambda z: z["slot_index"]):
        cid, pos, n = s["class_id"], s["class_position"], s["slot_index"] + 1
        if cid in kept:                        # already kept -> auto-collected IN PLACE. Still a real
            pending.append((n, color[cid]))    #   draw: the model is TOLD (below), so it observes the
            continue                           #   full T-length stream -- OBSERVATION PARITY with the
                                                #   tool player (no removal, no gappy < T view).
        if budget_left == 0:                   # no keeps left -> no decisions remain; tally the rest
            break
        pre = ""
        if pending:                            # report the kept-color draws that occurred since the
            evs = "; ".join(f"draw {dn}: another {col} {item} (already kept -> auto-collected, +1)"
                            for dn, col in pending)
            pre = f"(Since your last choice: {evs}.) "
            pending = []
        user = (f"{pre}Draw {n} of {T}: {_art(color[cid])} {color[cid]} {item} appears. "
                f"You have {budget_left} keep(s) left. KEEP or PASS?")
        messages.append({"role": "user", "content": user})
        chat_kwargs = {"max_tokens": 512, "temperature": temperature}
        if reasoning_effort is not None:
            chat_kwargs["reasoning_effort"] = reasoning_effort
        reply, exc = await call_with_retry(
            lambda: client.chat(model, system, messages, **chat_kwargs))
        if exc is None:
            u = dict(client.last_usage or {})
        else:
            # Transport failure surviving retries -- NOT a model decision. Flag distinctly from a
            # genuine parse failure ("default") rather than injecting a "DECISION: PASS" string,
            # which `parse_decision` used to read as an ordinary tag-parsed PASS, invisible in the
            # unparsed count (2026-07-09 framing-ladder strengthening fix).
            reply, u = f"[error {type(exc).__name__}, retries exhausted]", {}
        turn_usages.append(u)
        messages.append({"role": "assistant", "content": reply})
        if exc is None:
            dec, how = parse_decision(reply)
            if how == "default":                   # no KEEP/PASS token at all -> true parse failure
                unparsed += 1
        else:
            dec, how = "PASS", "error"
            unparsed += 1
        transcript.append({"slot": s["slot_index"], "color": color[cid], "class_id": cid,
                           "class_position": pos, "prompt": user, "decision": dec, "how": how,
                           "reply": reply,
                           "response_model": getattr(client, "last_response_model", None)})
        if dec == "KEEP":
            kept[cid] = pos
            budget_left -= 1
        if stop_after_turn is not None:
            stop_reason = stop_after_turn(turn_usages)
            if stop_reason and (budget_left > 0 or stop_reason == "missing_usage"):
                safety_stop = stop_reason
                break
    # tally any remaining draws of colors kept before budget ran out (loop `continue`d past them
    # only while budget>0; count draws that occur AFTER budget exhaustion too)
    collected = sum(1 for s in slots if s["class_id"] in kept and s["class_position"] >= kept[s["class_id"]])

    termination = safety_stop or ("budget_exhausted" if budget_left == 0 else "stream_complete")
    return {"kept": kept, "collected": collected, "budget": B, "unparsed": unparsed,
            "turn_usages": turn_usages, "color_of_class": color, "transcript": transcript,
            "messages": messages, "termination": termination}


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
    row = await run_episode(client, model, slots, T=T, B=B, system=SYSTEMS[vocab_key],
                            temperature=_ARGS.temp, item=vocab["item"], palette=vocab["palette"])
    row = {"seed": seed, "model_key": MODEL_KEY, "vocab": vocab_key, **row}
    (d / "session.json").write_text(json.dumps(row, indent=2))
    return cost_of(row["turn_usages"]), "ran"


async def main():
    load_dotenv()
    model = MODEL_STR
    for vk in VOCAB_KEYS:
        base_dir(vk).mkdir(parents=True, exist_ok=True)
    client = RawChat()
    cumulative = 0.0; inflight = 0; idx = 0; paused = False; lock = asyncio.Lock()
    UNITS = [(vk, seed) for vk in VOCAB_KEYS for seed in SEEDS]   # vocab-major: finish one vocab's report early

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

    print(f"URN abstraction-gap control: {MODEL_KEY}, uniform N={N}, g={G}, {len(SEEDS)} seeds x "
          f"{len(VOCAB_KEYS)} vocab(s) {VOCAB_KEYS} (clear labels, cap=${CAP_USD}) ...", flush=True)
    await asyncio.gather(*(worker() for _ in range(CONC)))
    print(f"\n==== {'PAUSED' if paused else 'COMPLETED'}: urn spend ${cumulative:.2f} ====", flush=True)
    report()


def report():
    report_summary(VOCAB_KEYS, base_dir, SEEDS, model_key=MODEL_KEY, pair_tool=PAIR_TOOL)


if __name__ == "__main__":
    asyncio.run(main())
