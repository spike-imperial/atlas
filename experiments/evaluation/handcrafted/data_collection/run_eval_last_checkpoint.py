"""
Runs an evaluation of the last checkpoints of all runs. The number of rollouts
per problem is 10. The size of the grid and the number of states is changed since
some problems in the set are o.o.d. (but that is OK since the model and the
observations are agnostic to that, except for the vanilla case).

Meanings:
- seq: sequential sampler
- rw: random walk sampler
- dr: domain randomization
- plr: robust prioritized level replay
- accel_full: accel using DR sampling from the full distribution (several objects, rooms, transitions)
- accel_scratch: accel using DR sampling from a simple distribution (one object, one room, one transition) to build from using mutations
- vanilla: not using GNN conditioning
- n*: using * edits (e.g., n20 -> using n20 edits). The default was uniformly sampled 7-10
- no-hind: no hindsight edits
- ftf, ftt: whether the level edits are on/off, HRM edits are on/off, hindsight edits are on/off
- pvl: using PVL scoring function instead of MaxMC
"""
import subprocess

ENTITY = None  # TODO: fill in your W&B entity here

RUNS = {  # TODO: fill in your W&B run IDs here
    # Sequential
    "seq-dr-i": [None] * 5,
    "seq-dr-c": [None] * 5,
    "seq-plr-i": [None] * 5,
    "seq-plr-c": [None] * 5,
    "seq-accel_full-i": [None] * 5,
    "seq-accel_full-c": [None] * 5,
    "seq-accel_scratch-i": [None] * 5,
    "seq-accel_scratch-c": [None] * 5,
    
    # Random Walk
    "rw-dr-i": [None] * 5,
    "rw-dr-c": [None] * 5,
    "rw-plr-i": [None] * 5,
    "rw-plr-c": [None] * 5,

    # Sequential (Ablations)
    "seq-plr-vanilla-i": [None] * 5,
    "seq-plr-myopic-i": [None] * 5,
    "seq-plr-domain_indep-i": [None] * 5,
    "seq-accel_full-n20-i": [None] * 5,
    "seq-accel_full-n3-i": [None] * 5,
    "seq-accel_full-n1-i": [None] * 5,
    "seq-accel_scratch-n20-i": [None] * 5,
    "seq-accel_scratch-n3-i": [None] * 5,
    "seq-accel_scratch-n1-i": [None] * 5,
    "seq-accel_scratch-no_hind-i": [None] * 5,
    "seq-accel_full-ftf-i": [None] * 5,
    "seq-accel_full-ftt-i": [None] * 5,
    "seq-accel_full-tft-i": [None] * 5,
    "seq-accel_full-tff-i": [None] * 5,
    "seq-accel_full-ttf-i": [None] * 5,
    "seq-plr-pvl-i": [None] * 5,
    "seq-plr-pvl-c": [None] * 5,
    "seq-accel_full-pvl-i": [None] * 5,
    "seq-accel_full-pvl-c": [None] * 5,
    "seq-accel_scratch-pvl-i": [None] * 5,
    "seq-accel_scratch-pvl-c": [None] * 5,
} 

for alg_name, alg_runs in RUNS.items():
    for r in alg_runs:
        run_name = f"handcrafted__{alg_name}__{r}"
        cmd = [
            "python", "scripts/eval_run.py",
            "--entity", ENTITY,
            "--eval_run_name", run_name,
            "--eval_file_cfg", "problems/03-hand-designed/eval.yaml",
            "--num_rollouts", "10",
            "--download",
            "--run_id", str(r),
            "--seed", "0",
            "--group", "eval-handcrafted-set-150",
            "--env_params", '{"height":19}',
            "--hrm_args", '{"max_num_states": 10,"max_num_literals":5}'
        ]
        subprocess.run(cmd)
