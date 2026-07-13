# Class-Bound R3 Protocol

## Status

Implementation and preflight validation are complete for benchmark version
`class-bound-r3-v1`. Publication collection has not started. The publication
runner refuses canonical seeds 2000--2023 unless the operator supplies the
explicit `--publication` flag.

Historical R3 artifacts use the global-primitive protocol and remain separate.
R0, R1, R2, R2c, and the economic-surface artifacts are unchanged.

## Revised contract

The environment attaches each successful `write_script` call to the hidden
`class_id` of the problem currently shown. The resulting script can execute on
that problem and on later problems only when the current hidden `class_id`
matches. `list_scripts` and `read_script` remain global and do not reveal a
script's binding.

The model receives this non-prescriptive disclosure:

> Each script is bound to the hidden problem type on which it is written. It
> may be run on the current problem and on later problems only when they have
> that same hidden type. Problem types remain hidden and unlabeled.

An invalid `run_script` attempt returns a neutral different-type error. It
does not execute the code, count as script use, or consume write budget. The
attempt is recorded in `n_cross_class_run_refused`.

Every successful `write_script` consumes one of the three writes, including a
replacement of an existing name. Replacements create a new artifact generation
and bind that generation to the current class.

## Prompt comparability

The historical R3 prompt said scripts persisted across every problem and the
tool schema expressly allowed general primitives and chaining. That protocol
permitted one script to serve multiple hidden classes.

R2c already defines a solver as applying to the current problem's hidden type
and all future problems of that same type. Class-bound R3 therefore repairs the
allocation-unit mismatch with R2c. R3 deliberately continues to differ from
R2c in execution, correctness-dependent reward, optional hand solving, manual
reuse, and debugging.

Class-bound R3 is a new protocol. Its results must not be pooled with or treated
as direct replications of historical global-primitive R3 sessions.

## Hidden data and artifacts

`slots_to_problems` carries `_class_id` and `_family` as server-only fields.
`problem_prompt` renders an allowlist and never exposes either field. Session
artifacts record:

- `benchmark_version: class-bound-r3-v1`
- `class_bound: true`
- the current script-to-class binding map
- one immutable artifact ID per successful write
- refused cross-class run attempts
- same-class persistence and beneficial-reuse counts by artifact generation

New output directories include `_class-bound-v1`. Smoke runs additionally use
an isolated `_smoke` directory. Existing directories are never overwritten.

## Publication experiments

All six panels use seeds 2000--2023, `N=8`, `T=60`, `B=3`, announced `N`, and
the canonical latent streams.

1. Haiku R3 repair: 24 repeat sessions with
   `claude-haiku-4-5-20251001`, magnitude 100.
2. Opus R3 repair: 24 repeat sessions with `claude-opus-4-8`, magnitude
   1000, canonical structural projection, and the Josephus sensitivity.
3. Qwen base R3 repair: 24 repeat sessions with
   `qwen-rl-base-q8:latest`, q8_0, and EFR4.
4. Qwen RL-final R3 repair: 24 repeat sessions with
   `qwen-rl-urn-final:latest`, q8_0, and EFR4.
5. GPT-5.4-mini R3 completion: 24 new sessions with
   `gpt-5.4-mini-2026-03-17`, `reasoning_effort=none`, `tool_choice=auto`,
   serial execution, and explicit global/per-seed cost breakers.
6. GPT-5.6 Sol R3 completion: 24 new sessions with `gpt-5.6-sol` and the
   same locked OpenAI controls.

Total planned collection: 144 sessions, comprising 96 repaired endpoints and
48 new GPT endpoints.

## Preflight evidence

The deterministic regression suite covers prompt/schema disclosure, hidden
label non-disclosure, same-class execution, cross-class refusal, replacement
charging and rebinding, legacy generic-session compatibility, artifact-level
scoring, and early stopping after three writes.

Canonical structural projections match 24/24 Haiku streams and 24/24 Opus
streams. Dry-run routing passed for all six model configurations without
network calls.

One noncanonical Haiku smoke session used seed 2999 in the isolated smoke
directory. It completed after three writes for $0.027673. On its third problem,
Haiku attempted to run an LCG script on a modular-exponentiation problem; the
harness refused the call, logged one cross-class attempt, and Haiku then wrote
a new script. This directly exercises the repaired boundary.

## Collection boundary

No canonical command should be executed until publication collection receives
separate approval. Every approved command must include `--publication`, an
explicit seed batch, and model-appropriate spend controls. GPT panels remain
serial and approval-gated by batch. The runner's default budget-exhaustion stop
preserves build timing; any reuse or correctness summary must state whether it
is observed only before truncation or evaluated separately on the untouched
tail.
