"""
Evaluates the last checkpoint for the runs associated with sequential RMs.
Five different runs per combination are included. The evaluation is performed
against a level-conditioned sampled set.
"""

import os

ENTITY = None  # TODO: Replace with your entity name on W&B

RUNS = {
    "dr-i": [None] * 5,  # TODO: Replace None with the actual run IDs
    "dr-c": [None] * 5,
    "plr-i": [None] * 5,
    "plr-c": [None] * 5,
    "accel_full-i": [None] * 5,
    "accel_full-c": [None] * 5,
    "accel_scratch-i": [None] * 5,
    "accel_scratch-c": [None] * 5,
    "plr_vanilla-i": [None] * 5,
    "plr_myopic-i": [None] * 5,
}

CMD = """
python scripts/eval_run.py 
--entity {entity} 
--eval_run_name {run_name} 
--eval_file_paths problems/01-cvar-sequential  
--num_rollouts 10 
--download 
--run_id {r} 
--seed {seed} 
--group seq-cvar-eval-cond-set
"""

for alg_name, alg_runs in RUNS.items():
    for r in alg_runs:
        for seed in range(2):
            run_name = f"{alg_name}__{r}__s{seed}"
            os.system(CMD.format(entity=ENTITY, run_name=run_name, r=r, seed=seed).strip().replace('\n', ''))
