"""Roster for the Qwen3 (not 3.5) DENSE model size-sweep on the oracle env.
Imported by run_qwen3_dense_job.sh via --qwen3dense. Kept separate from
oracle_sweep_config.QWEN3_MODELS (which holds the Qwen3.5 roster)."""

# HF ids, smallest -> largest. All dense (no MoE). BF16 sizes (approx):
#   0.6B~1.2GB, 1.7B~3.4GB, 4B~8GB, 8B~16GB, 14B~28GB, 32B~64GB -- all fit H100 80GB.
QWEN3_DENSE_MODELS = [
    "Qwen/Qwen3-0.6B",
    "Qwen/Qwen3-1.7B",
    "Qwen/Qwen3-4B",
    "Qwen/Qwen3-8B",
    "Qwen/Qwen3-14B",
    "Qwen/Qwen3-32B",
]
