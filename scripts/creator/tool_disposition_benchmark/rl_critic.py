"""Per-decision value critic (docs/rl-ppo-credit-assignment-spec.md §3-4). The genuinely new component
that closes the credit-assignment gap diagnosed after 5 flat GRPO pilots (see the spec's §0): a tiny,
separate MLP baseline `V(s_t)` so PASS decisions -- whose direct reward is always 0 -- still get a
meaningful advantage relative to what keeping would have earned.

Framing decision (spec §3.2, explicit sign-off given 2026-07-08): the critic gets a 5th, PRIVILEGED
feature (`rate`, the color's true draw probability from hidden stream metadata) that the policy never
sees. This is a training-time-only variance-reduction input (standard asymmetric/centralized-critic
practice), not something fed to the LLM -- the policy still only ever sees the draw sequence. A
non-privileged (4-feature, `privileged=False`) rerun is the planned follow-up once RL is confirmed to
work at all with the privileged version; the `privileged` flag exists everywhere below specifically so
that comparison is a one-flag change, not a rewrite.

This module's pure-python pieces (`extract_features`, `returns_to_go`, `ReturnNormalizer`) have NO torch
dependency and are exercised by `_selftest` even where torch isn't installed (matches `rl_train.py`'s
convention of lazy `import torch` inside functions, so files stay importable/testable on boxes or CI
without a GPU/torch present); the MLP itself (`build_critic` and everything that calls it) needs torch
and is skipped by `_selftest` if unavailable.

  PYTHONPATH=. python -m scripts.creator.tool_disposition_benchmark.rl_critic   # self-test
"""
from __future__ import annotations

FEATURE_NAMES_FAIR = ["t_frac", "budget_frac", "class_position_frac", "unkept_frac"]
FEATURE_NAMES_PRIV = FEATURE_NAMES_FAIR + ["rate"]


def extract_features(slots: list[dict], transcript: list[dict], *, T: int, B: int, N: int,
                     privileged: bool) -> list[list[float]]:
    """Per-decision feature vectors, aligned 1:1 with `transcript` (== one entry per KEEP/PASS decision,
    same alignment `rl_reward.per_decision_rewards` uses). Four engineered features computed purely from
    state already available in `run_episode`'s loop, no new data collection:

      t_frac              = t["slot"] / T                    -- fraction of the T-draw stream elapsed
      budget_frac         = (B - kept_so_far) / B             -- fraction of keep-budget remaining
                             BEFORE this decision (kept_so_far excludes the current, not-yet-decided draw)
      class_position_frac = t["class_position"] / T           -- how many times this color has recurred
                             so far (this draw included)
      unkept_frac         = |{other colors seen so far, not yet kept}| / N -- competition for the budget

    Plus a 5th, privileged-only feature `rate` (spec §3.2): the color's true draw probability from
    `stream_builder.py`'s per-slot `rate` field (same value for every slot of a given class_id, hence
    `setdefault` below rather than overwrite)."""
    rate_of: dict[int, float] = {}
    if privileged:
        for s in slots:
            rate_of.setdefault(s["class_id"], s["rate"])

    kept_so_far: set[int] = set()
    seen_so_far: set[int] = set()
    out: list[list[float]] = []
    for t in transcript:
        cid = t["class_id"]
        seen_so_far.add(cid)
        unkept_seen = len((seen_so_far - kept_so_far) - {cid})
        feat = [t["slot"] / T, (B - len(kept_so_far)) / B, t["class_position"] / T, unkept_seen / N]
        if privileged:
            feat.append(rate_of[cid])
        out.append(feat)
        if t["decision"] == "KEEP":
            kept_so_far.add(cid)
    return out


def returns_to_go(rewards: list[float]) -> list[float]:
    """G_t = sum(rewards[t:]) -- the exact Monte Carlo return-to-go, no bootstrapping or discount (spec
    §4: this is GAE with lambda=1; episodes are short enough (3-13 decisions) that the exact sum is
    already low-variance, so the backward TD(lambda) recursion buys nothing extra here)."""
    out = [0.0] * len(rewards)
    running = 0.0
    for i in range(len(rewards) - 1, -1, -1):
        running += rewards[i]
        out[i] = running
    return out


class ReturnNormalizer:
    """Running mean/std of observed G_t (Welford's online algorithm -- exact, no stored history), warm-
    started across outer steps like the critic itself (spec §3.3). The critic regresses onto the
    normalized target; `unnormalize` converts its prediction back to interpretable 'balls' units so the
    advantage (G_t - V(s_t)) stays in the same units as the reward."""

    def __init__(self):
        self.n = 0
        self.mean = 0.0
        self.m2 = 0.0

    def update(self, xs: list[float]) -> None:
        for x in xs:
            self.n += 1
            d = x - self.mean
            self.mean += d / self.n
            self.m2 += d * (x - self.mean)

    @property
    def std(self) -> float:
        if self.n < 2:
            return 1.0
        var = self.m2 / (self.n - 1)
        return var ** 0.5 if var > 1e-9 else 1.0

    def normalize(self, x: float) -> float:
        return (x - self.mean) / self.std

    def unnormalize(self, x: float) -> float:
        return x * self.std + self.mean

    def state_dict(self) -> dict:
        return {"n": self.n, "mean": self.mean, "m2": self.m2}

    def load_state_dict(self, d: dict) -> None:
        self.n, self.mean, self.m2 = d["n"], d["mean"], d["m2"]


def build_critic(n_features: int):
    """4-5 features -> 2x64 ReLU -> 1 scalar V(s_t) (spec §3.1) -- a few thousand parameters, negligible
    next to the 14B policy. Lazy `import torch.nn` (module-level docstring explains why). Deliberately
    stays on CPU (default device, never `.to("cuda")`) -- `train_critic_step`/`critic_values` build their
    input tensors on the default device too, so as long as callers never move the model, the whole critic
    trains/predicts on CPU: trivially fast at this size and keeps VRAM free for the 14B policy."""
    import torch.nn as nn

    class Critic(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(n_features, 64), nn.ReLU(),
                nn.Linear(64, 64), nn.ReLU(),
                nn.Linear(64, 1),
            )

        def forward(self, x):
            return self.net(x).squeeze(-1)

    return Critic()


def train_critic_step(critic, optimizer, normalizer: ReturnNormalizer,
                      features: list[list[float]], returns: list[float]) -> float:
    """One MSE regression step: V(s_t) -> normalized G_t. `features`/`returns` are the FLATTENED
    per-decision arrays across an entire outer-step batch (all episodes concatenated) -- decisions, not
    episodes, are the critic's training examples, giving it far more datapoints per step than the policy
    sees episodes. Target normalization happens INSIDE this call (updates `normalizer` first) so callers
    never have to remember the ordering. Returns the normalized-space MSE loss for logging."""
    import torch

    normalizer.update(returns)
    x = torch.tensor(features, dtype=torch.float32)
    y = torch.tensor([normalizer.normalize(r) for r in returns], dtype=torch.float32)
    pred = critic(x)
    loss = torch.nn.functional.mse_loss(pred, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()


def critic_values(critic, normalizer: ReturnNormalizer, features: list[list[float]]) -> list[float]:
    """V(s_t) in 'balls' units (un-normalized), no_grad -- used to FORM advantages (G_t - V(s_t)), never
    to train the critic itself (that's `train_critic_step`)."""
    import torch

    with torch.no_grad():
        x = torch.tensor(features, dtype=torch.float32)
        pred = critic(x).tolist()
    return [normalizer.unnormalize(p) for p in pred]


def save_critic(critic, normalizer: ReturnNormalizer, path) -> None:
    import torch
    torch.save({"state_dict": critic.state_dict(), "normalizer": normalizer.state_dict()}, path)


def load_critic(critic, normalizer: ReturnNormalizer, path, device) -> None:
    import torch
    ck = torch.load(path, map_location=device)
    critic.load_state_dict(ck["state_dict"])
    normalizer.load_state_dict(ck["normalizer"])


def _selftest():
    import statistics as st

    from scripts.creator.tool_disposition_benchmark.pi_star import eager_builds, wait_k_builds
    from scripts.creator.tool_disposition_benchmark.rl_reward import (
        builds_to_transcript, per_decision_rewards)
    from scripts.creator.tool_disposition_benchmark.stream_builder import (
        StochasticStreamSpec, build_stochastic_stream)
    from scripts.creator.tool_disposition_benchmark.urn_session import B, MAG, N, T, UNIFORM

    slots, _meta = build_stochastic_stream(StochasticStreamSpec(
        families=UNIFORM, n_hot=B, T=T, budget=B, guarantee_trap_early=1.0, magnitude=MAG, seed=2000))

    # --- returns_to_go: exact suffix sums, sanity-checked against a hand-picked short array
    rewards = [0.0, 5.0, 0.0, 0.0, 3.0]
    G = returns_to_go(rewards)
    assert G == [8.0, 8.0, 3.0, 3.0, 3.0], G
    assert G[0] == sum(rewards)      # G_0 always equals the episode's total scalar reward

    # --- ReturnNormalizer: matches plain mean/(sample-)std after a batch update
    norm = ReturnNormalizer()
    xs = [1.0, 2.0, 3.0, 4.0, 10.0]
    norm.update(xs)
    assert abs(norm.mean - st.mean(xs)) < 1e-9
    assert abs(norm.std - st.stdev(xs)) < 1e-9
    y = norm.normalize(xs[0])
    assert abs(norm.unnormalize(y) - xs[0]) < 1e-6

    # --- extract_features: shape, alignment, and the privileged/fair feature-count contract
    for name, builds in [("eager", eager_builds(slots, B)), ("wait2", wait_k_builds(slots, B, 2))]:
        transcript = builds_to_transcript(slots, builds, B)
        rewards = per_decision_rewards(slots, transcript)
        assert len(transcript) >= 3, "need a non-trivial decision sequence to exercise features"

        fair = extract_features(slots, transcript, T=T, B=B, N=N, privileged=False)
        priv = extract_features(slots, transcript, T=T, B=B, N=N, privileged=True)
        assert len(fair) == len(priv) == len(transcript) == len(rewards)
        assert all(len(f) == 4 for f in fair), fair
        assert all(len(f) == 5 for f in priv), priv
        assert all(f[:4] == p[:4] for f, p in zip(fair, priv)), \
            "privileged features must be a strict extension of the fair ones, not a different encoding"
        for f in fair:
            assert all(0.0 <= v <= 1.0 + 1e-9 for v in f), f   # all 4 fair features are fractions in [0,1]
        # last decision of the episode must have the correct number of keeps already spent
        n_keeps = sum(1 for t in transcript if t["decision"] == "KEEP")
        assert n_keeps <= B
        print(f"[{name}] {len(transcript)} decisions, {n_keeps} keeps, "
              f"reward sum={sum(rewards):.0f}, G_0={returns_to_go(rewards)[0]:.0f}")

    print("rl_critic pure-python self-test OK (features/returns/normalizer)")

    # --- MLP round trip: only if torch is installed (this repo's boxes have it; local dev may not)
    try:
        import torch
    except ImportError:
        print("torch not installed -- skipping Critic MLP train/save/load check "
              "(runs on the GPU box where this module is actually used)")
        return

    import tempfile
    from pathlib import Path

    builds = eager_builds(slots, B)
    transcript = builds_to_transcript(slots, builds, B)
    rewards = per_decision_rewards(slots, transcript)
    features = extract_features(slots, transcript, T=T, B=B, N=N, privileged=True)
    returns = returns_to_go(rewards)

    critic = build_critic(n_features=5)
    optimizer = torch.optim.Adam(critic.parameters(), lr=1e-3)
    normalizer = ReturnNormalizer()
    losses = [train_critic_step(critic, optimizer, normalizer, features, returns) for _ in range(200)]
    assert losses[-1] < losses[0], (losses[0], losses[-1])   # loss must actually decrease with training
    values = critic_values(critic, normalizer, features)
    assert len(values) == len(returns)

    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "critic.pt"
        save_critic(critic, normalizer, path)
        critic2 = build_critic(n_features=5)
        normalizer2 = ReturnNormalizer()
        load_critic(critic2, normalizer2, path, device="cpu")
        values2 = critic_values(critic2, normalizer2, features)
        assert values == values2, "save/load round trip must reproduce identical predictions"

    print(f"rl_critic MLP self-test OK (loss {losses[0]:.4f} -> {losses[-1]:.4f} over 200 steps, "
          f"save/load round trip verified)")


if __name__ == "__main__":
    _selftest()
