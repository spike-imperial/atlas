import argparse

import pandas as pd
import wandb

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--entity", required=True)
    parser.add_argument("--project", required=True)
    args = parser.parse_args()

    api = wandb.Api()

    for group, filename in [
        ("seq-cvar-eval-cond-set", "cvar_seq.csv"),
        ("dags-cvar-eval-cond-set", "cvar_rw.csv")
    ]:
        data = []
        runs = api.runs(f"{args.entity}/{args.project}", filters={"group": group})
        for run in runs:
            experiment_name, training_run_id, seed = run.name.split("__")
            row = {
                "experiment": experiment_name,
                "training_run_id": training_run_id,
                "seed": seed,
            }

            for key, value in run.summary.items():
                *_, prob_name = key.split("/")
                if prob_name.isdigit():
                    row[prob_name] = value

            data.append(row)

        df = pd.DataFrame(data)
        df.to_csv(filename, index=False)
