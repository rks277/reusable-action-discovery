# Publication-grade Qwen tool-transfer rerun — Preregistration + Implementation Plan

**Status (2026-07-10): COMPLETE.** Run on A100-40GB `ubuntu@158.101.121.179`. Both arms at matched q8_0,
n=24 paired seeds (2000–2023), EFR4, per-tag calibrated `a_script` (base 0.89 / RL-final 0.94).
**Result: RL-final 95% first-sight in the tool frame (58/61) vs 32% in the urn, indistinguishable from
base (95%, 59/62); 0 malformed/unknown/refused and 0/24 token-cap hits across all 48 sessions; regret
1821±396 (RL) / 1909±417 (base), ~halved and de-biased vs the original run.** The transfer-failure verdict
holds confounder-clean. Record updated: `rl-phase1-results.md` §2/§6/§8, `paper-structure-outline.md` §8.4,
project-review canvas. This section is the plan that produced that result — converting the RL tool-transfer
result (`docs/rl-phase1-results.md` §2) from a *behavioral timing signal* into a fully citable cell via
matched quantization, calibrated `a_script`, robust empty-fence handling, and 24 paired seeds. It is one of the three inference-only publication gates in the project
review canvas (alongside the paired Opus tool cell and the cross-family breadth replication) and is
required by `docs/paper-structure-outline.md` §8.4.

**Resolved decisions (2026-07-10 conversation):** (1) **A100-40GB is sufficient** — this is 14B
*inference*, and the original §1–4 tool eval already ran on an A100-40GB; the "quantize more on A100"
constraint was for 32B (CPU-offload) and for RL *training* VRAM, neither of which applies here (§6).
(2) **Match at q8_0, not q4_0** — the retained urn RL-final number (32%) is q8_0, so reading the tool
side at q8_0 holds the policy's quant fixed *across the two frames being contrasted*; q4 would reinject
the exact confound T1 removes (§3, T1). (3) **Reuse the existing 12-seed q8_0 RL-final tool run**; only
the **base control is produced fresh at q8_0** and both arms extend to 24 seeds under EFR4 (§4.1, §9).

**No live multi-seed run until the §9 stop point is cleared — `[[no-auto-reps]]`.**

Companion docs: results being upgraded — `docs/rl-phase1-results.md`; run procedure —
`docs/box-setup.md` §B2/§B3; project framing/notation — `docs/online-tool-investment-plan.md` §0.

---

## 1. What is already solid, and what this rerun does NOT re-litigate

The disposition verdict — **a reward-learned reserve policy does not activate in reusable script
creation** — is already carried by the behavioral lead metrics and is *not* what this rerun exists to
establish:

- **First-sight % / lateness** (A2, N disclosed on both sides): RL-final **95%** first-sight in the
  tool frame vs **32%** in the urn — a ~63-point within-model gap. These metrics are computed only over
  realized builds at the moment of first sighting and are `a_script`-independent, so they are unaffected
  by every threat this rerun repairs (`rl-phase1-results.md` §3, "why both-eager is robust to the tail").
- **Channel legibility**: 0 malformed / 0 unknown / 0 refused tool calls for both models — the failure
  is not a channel artifact (`rl-phase1-results.md` §2 Stage 1).
- **The tail never masked transfer**: across every diagnostic arm (config levers, `--empty-fence-retry`,
  low temp), first-sight stayed ~91–100% (`rl-phase1-results.md` §4.1). This falsification test is done.

**This rerun therefore does not attempt to overturn the transfer verdict.** It closes the four
specific reviewer objections that currently make the *build-count and regret* numbers, and the overall
cell, not-yet-publication-grade.

## 2. The four threats this rerun closes (each with the exact current defect)

| # | Threat | Current defect (source) | Fix |
|---|---|---|---|
| T1 | **Quant mismatch** | base served **Q4** vs RL-final **q8_0** — the paired delta is confounded with quantization (`rl-phase1-results.md` §2 setup, §6 caveat) | Serve BOTH tags at the **same quant** (q8_0), both merged/converted through the identical `convert_hf_to_gguf.py` path (§4.1) |
| T2 | **Uncalibrated `a_script`** | the `qwen-rl-*` served tags are absent from `arm_a1_announce._A_SCRIPT`, so regret silently defaults `a_script=1.0` instead of the measured 0.83 (`arm_a1_announce.py` L78–80; `rl-phase1-results.md` §6 caveat 1) | Measure `a_script` for **each served tag** via `a0_oracle_gap`, register the values in `_A_SCRIPT`, re-derive regret (§4.2) |
| T3 | **Idle-tail bias on build-count/regret** | empty-`\`\`\`json\`\`\``-fence collapse depresses realized builds/seed and inflates truncation, so builds/seed 1.75 and regret ~3.7k are truncation-biased (`rl-phase1-results.md` §4, §4.1) | Run with `--empty-fence-retry 4` (prune_no_tool; roughly halves regret, lifts realized builds) and report the empty-fence rate as a mechanical-validity field (§4.3) |
| T4 | **n = 12 seeds** | tool eval is noisy at n=12; the fresh base control read 100% vs the historical 88% first-sight (`rl-phase1-results.md` §6 caveat 4) | **24 paired seeds minimum** (2000–2023), 48 (2000–2047) as the stretch target; report seed-clustered bootstrap CIs (§3, §5) |

## 3. Design (locked)

**Subject.** The **14B** RL-final checkpoint (`runs/rl_urn_pilot/checkpoint/adapter/`) vs the untouched
`qwen2.5-coder:14b` base control. 14B is fixed by construction: it is the checkpoint that learned reserve
in the urn, so the transfer test *must* be on 14B. (32B is **not** part of this rerun — it would require
retraining a new subject; it is filed separately as the "bigger Qwen" robustness item,
`rl-phase1-results.md` §7 item 3.)

**Frame & information.** Paired tool A2, `arm_a1_announce.py --announce-n` — N disclosed on both sides,
matching the urn eval's information condition so the gap isolates framing, not N-ignorance
(`online-tool-investment-plan.md` §2). Same harness, same uniform-hard N=8 / g=1 / T=60 / B=3 pool.

**Pairing & seeds.** Same seed set for base and RL (same-seed → the delta is the policy, not stream
luck). **24 seeds (2000–2023)** minimum; **48 (2000–2047)** stretch. All eval seeds disjoint from
training seeds (9000+). Report every seed-level point.

**Serving config.** Both tags at `num_ctx 8192`, stock qwen2.5-coder tool template (so the paired delta
is weights, not context or template — as in the original §2), eval temperature = harness default (temp 0,
matching every prior eval and the pre-FT baseline). `tool_choice` unchanged from the current harness path.

**Primary vs secondary outcomes** (pre-committed):
- **Primary (verdict-bearing):** first-sight % and mean lateness, over realized builds. These are
  `a_script`- and tail-robust and already carry the claim; the rerun confirms they hold under matched
  quant and larger n.
- **Secondary (what the rerun makes citable):** builds/seed, regret vs π\* at the *calibrated* `a_script`,
  built-on-recurrence count, sustained-engagement / last-answered-problem.
- **Mechanical-validity (governs stop/go, never the verdict):** malformed / unknown / refused counts,
  unresolved-decision rate, empty-fence rate, token-cap-hit rate, stream-hash match.

## 4. Implementation

### 4.1 Matched-quant serving (T1)

Both tags must be produced through the *same* merge→GGUF path (`box-setup.md` §B2), quantized to
**q8_0**, num_ctx pinned to 8192. Base id (from the adapter config) is `Qwen/Qwen2.5-Coder-14B-Instruct`;
the base control must be that **same** bf16 base so base/RL differ only in the LoRA weights:

1. **RL-final (q8_0):** merge the local adapter `runs/rl_urn_pilot/checkpoint/adapter/` (275 MB, present)
   via `merge_lora.py --adapter … --out …/merged`, `convert_hf_to_gguf.py --outtype q8_0`, create
   `qwen-rl-urn-final` with `PARAMETER num_ctx 8192` off the stock qwen2.5-coder modelfile. (Exactly the
   §5 re-merge recipe already validated on the fresh box.)
2. **Base control (q8_0), the T1 fix:** save `Qwen/Qwen2.5-Coder-14B-Instruct` in bf16 (with the adapter
   dir's tokenizer/template, so both tags render identically) and convert to q8_0 via the **identical**
   `convert_hf_to_gguf.py --outtype q8_0`; create `qwen-rl-base-q8` with the same `num_ctx 8192` modelfile.
   This replaces the old `qwen-rl-base-ctx8k` (Q4) control.
3. Smoke-test one seed per tag (`box-setup.md` §A4): expect a clean report, 0 malformed calls.

**Reuse accounting (locked).** There is **no 24-seed tool run at any quant** — every existing tool run is
12 seeds (verified on disk 2026-07-10); the 24-seed runs on disk are all urn-side/other-task. So both arms
are extended to 24 fresh under EFR4 regardless. The existing 12-seed q8_0 RL-final run
(`runs/arm_a1_announce_qwen-rl-urn-final_latest_n-announced/`, no-EFR) is retained as a cross-check for the
tail-robust **primary** metrics only; the citable **secondary** (build-count/regret) numbers come from the
new EFR4 pass so all 24 seeds share one session-handling setting. Matching at q8 (not q4) is what lets the
RL-final artifact and the retained urn number stay on a single quant; q4 would force re-quantizing and
re-running RL-final and reinject a cross-frame quant difference.

### 4.2 `a_script` calibration (T2)

1. Run `a0_oracle_gap` on the **served q8_0 tags** (not just stock 14B) at MAG=100 over the benchmark
   families, e.g. `--models qwen-rl-urn-final:latest qwen-rl-base-q8:latest --magnitudes 100 --k 8`
   (run one model per grid per its own warning). Record pooled `a_script` per tag.
2. Register the measured values in `arm_a1_announce._A_SCRIPT` keyed on the exact served model strings, so
   the regret block (L183–206) uses the per-tag measured `a_script` instead of the 1.0 default.
   **Measured 2026-07-10 (q8_0, MAG=100, k=8, a_hand=0 everywhere): `qwen-rl-base-q8:latest` = 0.89,
   `qwen-rl-urn-final:latest` = 0.94** — both above the historical Q4 0.83 (q8 is closer to bf16), and
   RL-final > base, so per-tag pricing matters. Registered in `_A_SCRIPT` and verified on the box.
3. Re-derive regret for both models against the same exact-DP π\* (shared utility function, so the
   same-info reference is priced with the same `a_script`).

> Note: absolute regret is still the project's *secondary/noisy* metric; T2 makes base-vs-RL regret
> internally honest and the level comparable to the historical 2934±324 baseline (which used 0.83). It
> does not promote regret to a lead metric — lateness/first-sight remain the verdict.

### 4.3 Robust session handling (T3)

- Run with `--empty-fence-retry 4` (`arm_a1_announce.py` L39–43 → `driver.run_session`
  `prune_no_tool`/`max_no_tool_retries`): a no-tool turn is pruned from context and the same problem is
  re-prompted up to 4 times before force-advancing, so empty fences never accumulate and self-reinforce.
  This is the recommended publication lever (`rl-phase1-results.md` §4.1(b): breaks the runaway loop,
  ~halves regret, lifts realized builds/seed).
- Emit the **empty-fence rate**, **turns/seed**, and **last-answered-problem** per seed as reported fields
  so the residual tail is disclosed rather than hidden. EFR writes to a `_efr4`-suffixed dir, so it does
  not collide with baseline runs.
- **Honesty constraint (do not overclaim):** EFR is partial — engagement stays shallow (last-answered
  ~2–4 of 60) and correct-answers/seed did not rise in the diagnostic. Report builds/seed and regret as
  "de-biased but not fully tail-free on 14B"; a fully non-degenerate long-session tool eval still points
  to 32B or a tool-frame-trained model, which is out of scope here.

### 4.4 Sample size & statistics (T4)

- 24 paired seeds minimum (2000–2023), 48 stretch. Report: paired seed-level first-sight/lateness diffs,
  seed-clustered bootstrap CIs, and numerator/denominator for every percentage (per
  `paper-structure-outline.md` §4.4 statistical-reporting rules).
- Do **not** treat turns or per-problem decisions as independent observations; the seed is the unit.

## 5. Mechanical-validity gates (govern stop/go; behavioral outcomes never trigger a rerun)

Per the project's standing rule (`economic-response-surface-spec.md` §4; `paper-structure-outline.md`
§4.3): only *mechanical* failures may trigger a rerun; behavioral results may not.

- `assert_canonical`/stream-hash match on every seed (byte-identical streams across base/RL).
- 0 malformed / 0 unknown / 0 refused tool calls (any nonzero → investigate the channel, not the result).
- Unresolved-decision rate reported; pre-register a threshold (**>10% invalidates a seed**, matching the
  economic-surface rule).
- Token-cap-hit rate reported; a seed pinned at the 8192 ceiling with a front-truncated system prompt is a
  mechanical failure, not a behavioral pass.
- Distinct labels for transport errors vs model-decision failures (`call_with_retry` convention).

## 6. Run procedure (grounded in `box-setup.md`)

1. Stand up an A100/H100 box: Ollama + concurrency override (§A1), pull `qwen2.5-coder:14b` (§A2),
   sync repo/venv/.env (§A3) with the merge/GGUF deps (`peft accelerate bitsandbytes gguf sentencepiece
   protobuf`) and `llama.cpp` (§B3 step 2).
2. Re-pull the adapter and produce **both q8_0 tags** at `num_ctx 8192` (§4.1). Smoke-test each (§A4).
3. Calibrate `a_script` on both served tags (§4.2); register in `_A_SCRIPT`.
4. **Two-seed mechanical smoke** across both models with `--empty-fence-retry 4`, explicitly checking
   token-cap truncation and empty-fence rate (the exact modes that cost debugging cycles before).
5. Full paired run, 24 seeds (48 stretch), both models, `--announce-n --empty-fence-retry 4`.
6. **`rsync runs/ back before releasing the box`** — the standing lesson-learned rule (§A6/§B). The
   original Qwen A2 raw dirs were lost this way; do not repeat it.

**Cost/compute:** local inference is free (no per-token cost); the cost is box wall-clock. Estimate:
merge+convert+serve ≈ 15 min/tag; a_script calibration a few min/tag; paired 24-seed run ≈ tens of
minutes with EFR (turns/seed drops under EFR). Budget one short A100 session.

**Stop point for this pass: end of §6 step 3 (both tags served + `a_script` registered). No multi-seed
live run without separate approval (`[[no-auto-reps]]`).**

## 7. Outputs & how the record changes

- **Update `rl-phase1-results.md` §2** — replace the base(Q4)/RL(q8_0), n=12, `a_script=1.0` headline
  table with the matched-q8_0, n≥24, calibrated-`a_script`, EFR4 table. Keep the urn contrast column.
- **Update the §6 caveats** — retire caveats 1 (a_script), 3 (quant), and 4 (n=12) as resolved; keep the
  residual-tail caveat downgraded to "de-biased, not fully tail-free on 14B".
- **`paper-structure-outline.md` §8.4** — replace the placeholder ("same checkpoint remains 95%
  first-sight … replace these values") with the final numbers; feed Figure 4's reusable-script panel.
- **Canvas** (`tool-investment-project-review.canvas.tsx`) — flip the "Publication-grade Qwen
  tool-transfer rerun" row from "Publication required" to "Done" once §7 updates land.

## 8. Pre-committed interpretation (write before the numbers exist)

- **If the primary metrics hold** under matched quant + n≥24 + EFR (RL-final stays materially more eager
  in the tool frame than in the urn, ≳50-point first-sight gap; base stays open-loop): the transfer-failure
  cell is now citable — report the de-biased builds/seed and calibrated regret as the secondary evidence.
- **If matched quant / larger n materially moves first-sight** (e.g. RL-final drops well below 95% in the
  tool frame, or the base/RL gap collapses): report *that* honestly. The behavioral verdict currently rests
  on the confounded Q4-vs-q8_0 pairing at n=12; the whole point of the rerun is to be bound by whatever the
  hardened numbers say. Do not re-run to chase the prior result.
- **Language discipline** (`paper-structure-outline.md` writing rules): "matches the specified Bayesian
  comparator" not "optimal"; lead with timing, not regret; "proof-of-possibility RL case study"; never
  "the framing wall is absolute" (the §5.2 isomorphic keep/pass modality result already shows partial
  transfer).

---

## 9. Execution runbook (A100-40GB, locked 2026-07-10)

Grounded in `box-setup.md` §A/§B2/§B3. Replace `$BOX` with `ubuntu@<ip>`. Local artifacts confirmed
present: adapter (`runs/rl_urn_pilot/checkpoint/adapter/`, 275 MB) and the reusable 12-seed q8_0
RL-final tool run. **Stop at step 4 for approval before any multi-seed live run.**

### Step 0 — box up (from laptop)
```bash
# Ollama + concurrency override, pull the base, sync repo/venv/.env  (box-setup §A1–A3)
# then add the merge/GGUF deps + llama.cpp on top of the harness deps (box-setup §B3):
ssh $BOX 'cd ~/reusable-action-discovery && source .venv/bin/activate && \
  pip install -q torch transformers peft accelerate datasets bitsandbytes gguf sentencepiece protobuf && \
  git clone --depth 1 https://github.com/ggerganov/llama.cpp'
# re-sync the harness files edited locally (raw_chat.py fixes, arm_a1_announce EFR flag):
rsync -az --exclude=.git --exclude=runs --exclude=figs --exclude=.venv ./ $BOX:~/reusable-action-discovery/
# copy up the adapter (not in the excluded-runs sync):
rsync -az runs/rl_urn_pilot/checkpoint/adapter/ $BOX:~/reusable-action-discovery/runs/rl_urn_pilot/checkpoint/adapter/
```

### Step 1 — produce both q8_0 tags at num_ctx 8192 (T1)
```bash
ssh $BOX 'cd ~/reusable-action-discovery && source .venv/bin/activate && \
  # RL-final: merge adapter -> q8_0 GGUF -> ollama tag
  PYTHONPATH=. python -m scripts.creator.tool_disposition_benchmark.merge_lora \
    --adapter runs/rl_urn_pilot/checkpoint/adapter --out runs/rl_urn_pilot/merged --device auto && \
  python llama.cpp/convert_hf_to_gguf.py runs/rl_urn_pilot/merged --outfile ~/rl-urn-final-q8_0.gguf --outtype q8_0 && \
  # base control: SAME bf16 base (no adapter), same tokenizer/template, same convert path -> q8_0
  python -c "import torch;from transformers import AutoModelForCausalLM,AutoTokenizer;\
AutoModelForCausalLM.from_pretrained(\"Qwen/Qwen2.5-Coder-14B-Instruct\",dtype=torch.bfloat16).save_pretrained(\"runs/base_bf16\");\
AutoTokenizer.from_pretrained(\"runs/rl_urn_pilot/checkpoint/adapter\").save_pretrained(\"runs/base_bf16\")" && \
  python llama.cpp/convert_hf_to_gguf.py runs/base_bf16 --outfile ~/base-q8_0.gguf --outtype q8_0 && \
  # create both Ollama tags off the stock qwen tool template + num_ctx 8192
  ollama show --modelfile qwen2.5-coder:14b | grep -vE "^FROM |^# " > ~/tmpl.txt && \
  { echo "FROM /home/ubuntu/rl-urn-final-q8_0.gguf"; echo "PARAMETER num_ctx 8192"; cat ~/tmpl.txt; } > ~/final.modelfile && \
  { echo "FROM /home/ubuntu/base-q8_0.gguf";        echo "PARAMETER num_ctx 8192"; cat ~/tmpl.txt; } > ~/base.modelfile && \
  ollama create qwen-rl-urn-final -f ~/final.modelfile && ollama create qwen-rl-base-q8 -f ~/base.modelfile'
```

### Step 2 — smoke both tags (1 seed each; expect unparsed=0, 0 malformed)
```bash
ssh $BOX 'cd ~/reusable-action-discovery && source .venv/bin/activate && for m in \
  qwen-rl-base-q8:latest qwen-rl-urn-final:latest; do echo "##### $m #####"; timeout 300 env PYTHONPATH=. \
  python -u -m scripts.creator.tool_disposition_benchmark.arm_a1_announce \
    --model $m --announce-n --empty-fence-retry 4 --seeds 2000 --conc 1; done'
```

### Step 3 — calibrate a_script on BOTH served tags (T2)
```bash
ssh $BOX 'cd ~/reusable-action-discovery && source .venv/bin/activate && for m in \
  qwen-rl-base-q8:latest qwen-rl-urn-final:latest; do echo "##### a_script $m #####"; env PYTHONPATH=. \
  python -m scripts.creator.tool_disposition_benchmark.a0_oracle_gap \
    --models $m --magnitudes 100 --k 8; done'
```
Then register the measured pooled values in `arm_a1_announce._A_SCRIPT` (one line each, keyed on the exact
served strings, e.g. `"qwen-rl-urn-final:latest": 0.83, "qwen-rl-base-q8:latest": 0.83`) and re-sync the
edited file up. **← STOP HERE for approval (`[[no-auto-reps]]`).**

### Step 4 — full paired 24-seed run, EFR4, A2 (both arms)
```bash
ssh $BOX 'cd ~/reusable-action-discovery && source .venv/bin/activate && for m in \
  qwen-rl-base-q8:latest qwen-rl-urn-final:latest; do echo "##### $m #####"; nohup env PYTHONPATH=. \
  python -u -m scripts.creator.tool_disposition_benchmark.arm_a1_announce \
    --model $m --announce-n --empty-fence-retry 4 --seeds $(seq 2000 2023) --conc 4 \
    > ~/tool_rerun_$(echo $m | tr ":/" "__").log 2>&1; done & echo launched'
# watch: ssh $BOX 'tail -5 ~/tool_rerun_*.log'  (each seed prints first-sight/lateness + mechanical fields)
```

### Step 5 — sync back BEFORE releasing the box (the standing lesson)
```bash
rsync -az $BOX:~/reusable-action-discovery/runs/ ./runs/
```
Output dirs: `runs/arm_a1_announce_qwen-rl-{base-q8,urn-final}_latest_n-announced_efr4/` (24 seeds each).
Then do §7's record updates and re-derive regret at the registered `a_script`.

**Compute estimate:** merge+convert+serve ≈15 min/tag; a_script ≈3–5 min/tag; the paired 24-seed EFR4
run is tens of minutes (EFR lowers turns/seed). One short A100 session covers all of it; local inference
is free (no per-token cost), so the only budget is box wall-clock.
