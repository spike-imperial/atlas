import os
import sys

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def _plot_bars(df, algorithms, algorithm_names, colors, filename_prefix):
    for pname in problem_names:
        _, ax = plt.subplots(figsize=(1.3, 2.25))
        rows = []
        for algorithm, name in zip(algorithms, algorithm_names):
            df_alg_prob = df[df["experiment"] == algorithm][pname]
            for solve_rate_run in df_alg_prob.to_numpy():
                rows.append({
                    "algorithm": name,
                    "solve_rate": solve_rate_run
                })

        prob_df = pd.DataFrame(rows)
        sns.barplot(
            prob_df,
            y="algorithm",
            x="solve_rate",
            hue="algorithm",
            palette=colors,
            width=0.6,
            ax=ax,
        )

        # plt.title(pname)
        ax.set_xlabel("Solve Rate", fontsize="xx-large")
        ax.set_ylabel("")
        ax.set_xticks([0.0, 0.5, 1.0])
        ax.grid(True, alpha=0.2)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['left'].set_linewidth(2)
        ax.spines['bottom'].set_linewidth(2)
        ax.tick_params(length=0.1, width=0.1, labelsize="xx-large")
        ax.spines['left'].set_position(('outward', 10))
        ax.spines['bottom'].set_position(('outward', 10))
        ax.set_xlim(0, 1.04)
        name = f"{filename_prefix}-{pname}" if filename_prefix else pname
        plt.savefig(f"{name}.pdf", bbox_inches='tight', pad_inches=0.0)
        plt.close()


if __name__ == "__main__":
    set2 = sns.color_palette("Set2", 8)
    df = pd.read_csv(os.path.join(os.path.dirname(sys.argv[0]), "..", "data_collection", "eval_last_checkpoint.csv"))
    problem_names = df.columns.difference({"experiment", "training_run_id", "mean"})

    _plot_bars(
        df,
        algorithms=["seq-dr-i", "seq-plr-i", "seq-accel_full-i", "seq-accel_scratch-i"],
        algorithm_names=["DR", "PLR$^\\bot$", "ACCEL", "ACCEL-0"],
        colors=[set2[3], set2[0], set2[1], set2[2]],
        filename_prefix="seq-i"
    )

    _plot_bars(
        df,
        algorithms=["seq-dr-i", "seq-plr-i", "seq-accel_full-i", "seq-accel_scratch-i"],
        algorithm_names=["DR", "PLR$^\\bot$", "ACCEL", "ACCEL-0"],
        colors=[set2[3], set2[0], set2[1], set2[2]],
        filename_prefix="seq-c"
    )
