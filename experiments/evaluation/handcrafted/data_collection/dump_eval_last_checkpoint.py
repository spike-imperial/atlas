"""
Dumps the solve rate for each problem evaluated using the `run_eval_last_checkpoint` script.
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
    runs = api.runs(f"{args.entity}/{args.project}", filters={"group": "eval-handcrafted-set-150"})
    for run in runs:
        experiment_name, training_run_id = run.name.split("__")[1:]
        row = {
            "experiment": experiment_name,
            "training_run_id": training_run_id,
            "mean": run.summary.get("eval/episode_solve_rate/mean", 0.0)
        }

        for key, value in run.summary.items():
            if key.startswith("eval/episode_solve_rate/hc-"):
                row[key[len("eval/episode_solve_rate/hc-"):]] = value

        data.append(row)

    df = pd.DataFrame(data)
    df.to_csv("eval_last_checkpoint.csv", index=False)
