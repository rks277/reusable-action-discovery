"""Roster for the Qwen2.5 dense instruct size-sweep on the oracle env.
Imported by run_oracle_sweep via --qwen25. Qwen2.5 has NO thinking mode, so no
/no_think is applied (run_oracle_llm._sys only adds it for qwen3*)."""

# HF ids, smallest -> largest. Dense instruct. BF16 sizes (approx):
#   0.5B~1GB, 1.5B~3GB, 3B~6GB, 7B~15GB, 14B~28GB, 32B~64GB -- all fit H100 80GB.
QWEN25_MODELS = [
    "Qwen/Qwen2.5-0.5B-Instruct",
    "Qwen/Qwen2.5-1.5B-Instruct",
    "Qwen/Qwen2.5-3B-Instruct",
    "Qwen/Qwen2.5-7B-Instruct",
    "Qwen/Qwen2.5-14B-Instruct",
    "Qwen/Qwen2.5-32B-Instruct",
    # 72B: BF16 ~145GB -> needs 2 GPUs (tensor-parallel-size 2). Only run via the
    # dedicated run_qwen25_72b_job.sh; the single-GPU job's bash array excludes it.
    "Qwen/Qwen2.5-72B-Instruct",
]
