import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def plot_heatmap(
    df,
    rep_col="representation",
    model_col="model",
    value_col="MAE",
    figsize=(10, 8),
    cmap="viridis",
    fmt=".2f",
    highlight="col",  # options: "col", "row", "global", None
):

    # --- Style ---
    sns.set_theme(style="white")  # cleaner background

    # 1. Aggregate
    stats = (
        df.groupby([rep_col, model_col])[value_col]
        .agg(["mean", "std"])
        .reset_index()
    )

    # 2. Pivot
    mean_matrix = stats.pivot(index=rep_col, columns=model_col, values="mean")
    std_matrix = stats.pivot(index=rep_col, columns=model_col, values="std")

    # 3. Annotation
    annot = mean_matrix.copy().astype(str)

    for i in range(mean_matrix.shape[0]):
        for j in range(mean_matrix.shape[1]):
            m = mean_matrix.iloc[i, j]
            s = std_matrix.iloc[i, j]

            if pd.isna(m):
                annot.iloc[i, j] = ""
            elif pd.isna(s):
                annot.iloc[i, j] = f"{m:{fmt}}"
            else:
                annot.iloc[i, j] = f"{m:{fmt}} ± {s:{fmt}}"

    # 4. Plot
    fig, ax = plt.subplots(figsize=figsize)

    heatmap = sns.heatmap(
        mean_matrix,
        annot=annot,
        fmt="",
        cmap=cmap,
        linewidths=1,
        cbar=False,
        linecolor="white",
        square=True,
        cbar_kws={"label": f"Mean {value_col}"},
        annot_kws={
            "size": 15,  # 🔥 bigger text
            "weight": "bold",  # 🔥 bold text
            # "color": "black"
        },
        ax=ax,
    )

    # --- Axis styling ---
    # ax.set_title(f"{value_col} (mean ± std)", fontsize=16, weight="bold", pad=15)

    ax.set_xlabel(
        model_col.capitalize(), fontsize=18, weight="bold", labelpad=10
    )
    ax.set_ylabel(
        rep_col.capitalize(), fontsize=18, weight="bold", labelpad=10
    )

    # Tick labels
    ax.set_xticklabels(
        ax.get_xticklabels(), rotation=30, ha="right", fontsize=15
    )

    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=15)

    # Remove spines for cleaner look
    for spine in ax.spines.values():
        spine.set_visible(False)

    # Colorbar styling
    # cbar = heatmap.collections[0].colorbar
    # cbar.ax.tick_params(labelsize=11)

    # --- Highlight logic ---
    if highlight is not None:

        if highlight == "col":
            mask = mean_matrix.eq(mean_matrix.min(axis=0), axis=1)

        elif highlight == "row":
            mask = mean_matrix.eq(mean_matrix.min(axis=1), axis=0)

        elif highlight == "global":
            mask = mean_matrix == np.nanmin(mean_matrix.values)

        else:
            raise ValueError(
                "highlight must be one of: 'col', 'row', 'global', None"
            )

        # Draw rectangles around highlighted cells
        for i in range(mean_matrix.shape[0]):
            for j in range(mean_matrix.shape[1]):
                if mask.iloc[i, j]:
                    ax.add_patch(
                        plt.Rectangle(
                            (j, i),
                            1,
                            1,
                            fill=False,
                            edgecolor="black",  # 🔥 highlight color
                            lw=2.5,
                        )
                    )
    plt.tight_layout()
    return fig, ax
