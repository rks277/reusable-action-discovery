# mcp_creator — change log (read this if you're running the ladder)

## 2026-06-24 — submit_answers is now TERMINAL  ⚠️ AFFECTS COMPARABILITY

**What changed.** The episode now ends the instant the model makes its **first successful
`submit_answers`** call. No re-submits, no post-submit tool calls (previously: "last-wins,
submit anytime, episode runs until token cap"). Implemented in `driver.py` (breaks the loop
right after a successful submit); reflected in the model-facing docs:
- `prompts.py` system prompt: "submitting ENDS the episode immediately (no resubmits)…"
- `episode_state.py` `submit_answers` schema + `op_submit_answers` message: "This is FINAL…"

**Why.** Under the old behavior a model could submit a correct set early, then keep burning
budget (re-submitting, retrying write_script — which is refused post-begin_test — etc.) and
overshoot the token cap. That made `spent_tokens` a poor efficiency measure and produced
`hit_cap=True` episodes that had actually solved (e.g. Haiku items 8/0: 20/20 but 101k–106k
spent against a 100k cap, because the submit happened at tool-call #23 and 7 more calls
followed). Terminal submit makes `spent_tokens` a true cost-to-answer and removes the
post-submit noise.

**⚠️ Comparability.** Runs produced BEFORE this change used the old semantics (soft cap +
resubmits). In particular any earlier `runs/mcp_creator_*` episodes — including the 7B/14B
ladder numbers logged in the shared memory note — were under the old rules. **Do not pool
pre-change and post-change runs**, especially for `spent_tokens` / `hit_cap` / efficiency.
Solve/recognition metrics are largely unaffected (submission content is unchanged), but
`spent_tokens` will drop for fast solvers and `hit_cap` rates will fall. Re-run any rung you
need token-efficiency numbers for. Tag/segregate by date if mixing is unavoidable.

**Unchanged.** Phase gating (write-lock stays at `begin_test`, irreversible, prominently
documented), the I/O contract, the token-cap accounting (still cumulative input+output,
checked between turns so a crossing turn still completes — minor overshoot possible but now
rare since most episodes end at submit), neutral prompt, the 4-tool interface.
