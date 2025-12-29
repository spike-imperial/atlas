import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import bootstrap


def bootstrap_ci_bounded(data, bounds=(0, 1)):
    data_clean = data.dropna()

    if data_clean.std() == 0:
        value = data_clean.iloc[0]
        return value, value

    res = bootstrap((data_clean,), np.mean, confidence_level=0.95, random_state=42)

    # Actually clip to valid bounds
    low = max(res.confidence_interval.low, bounds[0])
    high = min(res.confidence_interval.high, bounds[1])
    return round(low, 2), round(high, 2)


def _make_latex_table(df, algorithms, names, caption, label):
    # Select the algorithms
    algorithms_df = df[df["experiment"].isin(algorithms)]

    # Make new table having the (algorithm, problem, solve_rate) structure
    melted_df = pd.melt(
        algorithms_df,
        id_vars=['experiment'],  # Column(s) to keep as identifier
        value_vars=algorithms_df.columns.difference({"experiment", "training_run_id", "mean"}),  # Columns to melt
        var_name='problem',  # Name for the new column containing column names
        value_name='solve_rate'
    )

    # Aggregate scores across algorithms and problems, and create string "mean (ci_lower, ci_upper)
    agg_df = melted_df.groupby(["experiment", "problem"])["solve_rate"].agg(
        {"mean", bootstrap_ci_bounded}).reset_index()
    agg_df['mean_ci'] = agg_df['mean'].round(2).astype(str) + ' ' + agg_df['bootstrap_ci_bounded'].astype(str)

    # Pivot the table and make final table
    pivoted_df = agg_df.pivot(
        index='problem',
        columns='experiment',
        values='mean_ci'
    )
    pivoted_df.index = [
        '$\\mathtt{{{}}}$'.format(prob.replace('_', '\\_')).replace('-', '\\text{-}')
        for prob in pivoted_df.index
    ]
    pivoted_df = pivoted_df[algorithms]
    pivoted_df = pivoted_df.rename(columns={k: v for k, v in zip(algorithms, names)})
    pivoted_df.index.name = "Problem"
    return pivoted_df.to_latex(
        longtable=True,
        caption=f'Solve rates and 95\% CIs for hand-designed problems with {caption}.',
        label=f'tab:per_problem_results_{label}'
    )


if __name__ == "__main__":
    path = os.path.dirname(sys.argv[0])
    csv_path = os.path.join(path, "..", "data_collection", "eval_last_checkpoint.csv")
    df = pd.read_csv(csv_path)

    with open(os.path.join(path, "seq_indep.tex"), 'w') as f:
        f.write(_make_latex_table(
            df,
            algorithms=["seq-dr-i", "seq-plr-i", "seq-accel_full-i", "seq-accel_scratch-i"],
            names=["DR", "PLR$^\\bot$", "ACCEL", "ACCEL-0"],
            caption="sequential task sampling and independent problem sampling",
            label="seq_indep"
        ))

    with open(os.path.join(path, "seq_cond.tex"), 'w') as f:
        f.write(_make_latex_table(
            df,
            algorithms=["seq-dr-c", "seq-plr-c", "seq-accel_full-c", "seq-accel_scratch-c"],
            names=["DR", "PLR$^\\bot$", "ACCEL", "ACCEL-0"],
            caption="sequential task sampling and level-conditioned problem sampling",
            label="seq_cond"
        ))

    with open(os.path.join(path, "rw.tex"), 'w') as f:
        f.write(_make_latex_table(
            df,
            algorithms=["rw-dr-i", "rw-plr-i", "rw-dr-c", "rw-plr-c"],
            names=["DR$_\\text{indep}$", "PLR$^\\bot_\\text{indep}$", "DR$_\\text{cond}$", "PLR$^\\bot_\\text{cond}$",],
            caption="random walk-based task sampling",
            label="rw"
        ))
