"""
Runs an evaluation for a series of checkpoints for all runs below. The
purpose is to produce a learning curve over the testing set. This is
costly, so it is only performed for a few runs.

See `eval_last_checkpoint.py` for details on the meanings of the runs.
"""
import subprocess

ENTITY = None  # TODO: fill in your W&B entity here

RUNS = {  # TODO: fill in your W&B run IDs here
    "seq-dr-i": [None] * 5,
    "seq-dr-c": [None] * 5,
    "seq-plr-i": [None] * 5,
    "seq-plr-c": [None] * 5,
    "seq-accel_full-i": [None] * 5,
    "seq-accel_full-c": [None] * 5,
    "seq-accel_scratch-i": [None] * 5,
    "seq-accel_scratch-c": [None] * 5,
}

for alg_name, alg_runs in RUNS.items():
    if alg_name.startswith("seq"):
        steps = [10, *range(100, 2001, 100)]
    else:
        steps = [10, *range(100, 3001, 100)]

    steps = [str(x) for x in steps]

    for r in alg_runs:
        run_name = f"{alg_name}__{r}"
        cmd = [
            "python", "scripts/eval_run.py",
            "--entity", ENTITY,
            "--eval_run_name", run_name,
            "--eval_file_cfg", "problems/03-hand-designed/eval.yaml",
            "--num_rollouts", "10",
            "--download",
            "--run_id", str(r),
            "--seed", "0",
            "--group", "eval-handcrafted-set-curves-150",
            "--env_params", '{"height":19}',
            "--hrm_args", '{"max_num_states": 10,"max_num_literals":5}',
            "--steps", *steps
        ]
        subprocess.run(cmd)
