"""
Evaluates the last checkpoint for the runs associated with DAG RMs.
Five different runs per combination are included. The evaluation is performed
against a level-conditioned sampled set.
"""

import os

ENTITY = None  # TODO: Replace with your entity name on W&B

RUNS = {
    "dr-i":  [None] * 5,  # TODO: Replace None with the actual run IDs
    "dr-c":  [None] * 5,
    "plr-i": [None] * 5,
    "plr-c": [None] * 5
}

CMD = """
python scripts/eval_run.py 
--entity {entity} 
--eval_run_name {run_name} 
--eval_files_path problems/02-cvar-dags 
--num_rollouts 10 
--download 
--run_id {r} 
--seed {seed} 
--group dags-cvar-eval-cond-set
"""

for alg_name, alg_runs in RUNS.items():
    for r in alg_runs:
        for seed in range(2):
            run_name = f"{alg_name}__{r}__s{seed}"
            os.system(CMD.format(entity=ENTITY, run_name=run_name, r=r, seed=seed).strip().replace('\n', ''))
