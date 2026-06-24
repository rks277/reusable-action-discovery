# tool-wood-discrepancy — Coupon-collector WoodWorld

Tests whether ToolWorld's **distinct goal items + collect-the-keys goal** (vs WoodWorld's
fungible "hold N identical wood") is the remaining structural reason recognition behaves
differently across the two worlds. We make WoodWorld a **coupon collector**: the goal is to
collect all **N distinct kinds** of wood — the direct analog of opening N distinct doors /
collecting N distinct keys in ToolWorld.

## Mechanics
- **Goal:** hold ≥1 of each of the N distinct wood kinds (`solved` = held all N).
- **`gather`:** drops 1 uniformly-random wood-kind (the coupon) **+** 1 uniformly-random
  tree-resource (one of T types). Both probability 1.
- **T tree-resources:** 1 latent **stick** + (T−1) inert **apples**, relabeled as a
  look-alike numbered family `{stem}{i}` (the agent must search which pair combines).
- **Build:** `stick + stick → axe` (non-hostile; sticks are free byproducts). Axe persists.
- **`use axe`:** yields 1 uniformly-random wood-kind **not yet held**, w.p. 1.

## Economics
- Grind (gather only) = coupon collector: `E[grind] = N·H_N`, `H_N = Σ 1/k`.
- **Budget** = `round(1.2 · N · H_N)` (1.2× the expected grind; ToolWorld slack regime).
- Build path (approx): `2T` gathers to get 2 sticks (coupons accrue meanwhile) +
  recipe search `~(C(T,2)+1)/2` + 1 craft + `~N·(1−1/N)^(2T)` axe-uses for the rest.

## Sweep
Haiku, **T = 2..8** (x) × **N = 2..20** (y), 1 rep/cell = **133 episodes**, nohint,
run-to-solve. `no_progress` safeguard is **disabled** (the coupon tail is legitimately
"no new progress" — duplicate gathers waiting for the last kind); the budget binds.

## Files
- `coupon_woodworld.py` — env (`make_world` / `State` / `run`); reuses `lomekwi.obfuscation.assign`,
  the woodworld action parser, and `RawChat`.
- `budget_coupon.py` — `H`, `expected_grind`, `budget_for_coupon`, `expected_build`.
- `run_coupon_sweep.py` — (T,N) sweep; mirrors `scripts/run_woodworld_region_sweep.py`.
- `plot_coupon_heatmaps.py` — recognition + solve (T,N) heatmaps.
- `runs/`, `figs/` — outputs.

## Run
```bash
# smoke (one T=8,N=20 cell)
PYTHONPATH=. python tool-wood-discrepancy/run_coupon_sweep.py --smoke
# full sweep (~10–20 min, ~$2–5 at Haiku)
PYTHONPATH=. python tool-wood-discrepancy/run_coupon_sweep.py --conc 12 --max-cost 10
# resume if interrupted
COUPON_RESUME=tool-wood-discrepancy/runs/<dir> PYTHONPATH=. python tool-wood-discrepancy/run_coupon_sweep.py
# plot
PYTHONPATH=. python tool-wood-discrepancy/plot_coupon_heatmaps.py tool-wood-discrepancy/runs/<dir>
```

## Episode row schema (episodes.jsonl)
`n_types`=T, `n`=`n_kinds`=N, `built_axe`, `held_ingredients` (ever held ≥2 sticks),
`solved`, `distinct_held`, `build_turn`, `t_star`, `use_axe_count`, `craft_attempts`,
`total_actions`, `stopped_reason`, `usage`, `labels`, `actions`/`agent_texts`/`obs`, `elapsed_s`.

## Edge cases
- **N=2** → budget = round(1.2·2·1.5) = 4: grind solves often, no room to build → bottom
  rows near-empty for recognition (expected, not a bug).
- **T=2** → stick is 1-of-2, builds easily → recognition high on the left (the contrast).
- Letter obfuscation caps at 26; N≤20 (+axe+stem) is safe (asserted in `make_world`).

## Read the result
Does recognition rise/fall with N and T like ToolWorld's Haiku panel (R≈0.86)? If the
coupon goal makes WoodWorld recognition behave ToolWorld-like, the distinct-goal-items axis
is the driver.
