"""
Dumps the solve rate for each problem evaluated using the `run_eval_checkpoint_seq` script.
This data can be used to build a curve with aggregate performance.
"""
import argparse

import pandas as pd
import wandb

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--entity", required=True)
    parser.add_argument("--project", required=True)
    args = parser.parse_args()

    api = wandb.Api()

    data = []
    runs = api.runs(f"{args.entity}/{args.project}", filters={"group": "eval-handcrafted-set-curves-150"})
    for run in runs:
        experiment_name, training_run_id = run.name.split("__")
        problem_metrics = [key for key in run.summary.keys() if key.startswith("eval/episode_solve_rate/hc-")]
        history = run.history(keys=problem_metrics)

        for step in history["_step"].values:
            row = {
                "experiment": experiment_name,
                "training_run_id": training_run_id,
                "step": step
            }
            step_row = history[history["_step"] == step]
            for m in problem_metrics:
                row[m[len("eval/episode_solve_rate/hc-"):]] = step_row[m].iloc[0]

            data.append(row)

    df = pd.DataFrame(data)
    df.to_csv("eval_checkpoint_seq.csv", index=False)
