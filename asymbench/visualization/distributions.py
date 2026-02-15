import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def plot_distribution_by_category(
    data: pd.DataFrame,
    x_col: str,
    y_col: str,
    order:list|None=None,
    figsize:tuple[int,int]=(10, 4),
    dpi:int=350,
    show_points:bool=True,
    point_alpha:float=0.35,
    title:str|None=None,
    color_map: dict | None = None,
):
    """
    Publication-quality violin plot:
    - x-axis: categorical column (x_col)
    - y-axis: distribution column (y_col)
    - ordered by increasing mean(y_col)

    Parameters
    ----------
    data : pd.DataFrame
    x_col : str
        Column used for x-axis categories
    y_col : str
        Column containing distribution values
    order : list or None
        Optional manual order of categories
    color_map : dict or None
        Mapping {category: color}. If None, default colors are used.
    """

    df = data[[x_col, y_col]].dropna().copy()

    # Order categories by increasing mean of y_col
    if order is None:
        order = (
            df.groupby(x_col)[y_col]
            .mean()
            .sort_values()
            .index
            .tolist()
        )

    groups = [df.loc[df[x_col] == cat, y_col].to_numpy() for cat in order]

    # --- Color handling ---
    if color_map is not None:
        cat_to_color = {
            cat: color_map.get(cat, "gray") for cat in order
        }
    else:
        colors = plt.cm.tab10.colors
        cat_to_color = {
            cat: colors[i % len(colors)]
            for i, cat in enumerate(order)
        }

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    # --- violin plot ---
    parts = ax.violinplot(
        groups,
        showmeans=False,
        showmedians=False,
        showextrema=False,
    )

    # Color each violin
    for i, body in enumerate(parts["bodies"]):
        cat = order[i]
        color = cat_to_color[cat]
        body.set_facecolor(color)
        body.set_edgecolor(color)
        body.set_alpha(0.25)
        body.set_linewidth(1.2)

    # --- draw mean bars manually ---
    for i, (cat, vals) in enumerate(zip(order, groups), start=1):
        if len(vals) == 0:
            continue
        mean_val = np.mean(vals)
        color = cat_to_color[cat]

        ax.plot(
            [i - 0.1, i + 0.1],
            [mean_val, mean_val],
            color=color,
            linewidth=2.2,
            solid_capstyle="round",
            zorder=4,
        )

    # --- scatter points ---
    if show_points:
        rng = np.random.default_rng(0)
        for i, (cat, vals) in enumerate(zip(order, groups), start=1):
            color = cat_to_color[cat]
            x = rng.normal(loc=i, scale=0.06, size=len(vals))
            ax.scatter(
                x,
                vals,
                s=14,
                alpha=point_alpha,
                color=color,
                edgecolors="none",
            )

    # --- axis styling ---
    if title:
        ax.set_title(title, fontsize=18, pad=10)

    ax.set_xticks(np.arange(1, len(order) + 1))
    ax.set_xticklabels(order, rotation=0, ha="right")
    ax.tick_params(axis="both", labelsize=14)

    ax.set_xlabel(x_col.replace("_", " ").title(), fontsize=18)
    ax.set_ylabel(y_col.upper(), fontsize=18)

    ax.grid(True, axis="y", alpha=0.5)

    # Clean spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    return fig, ax