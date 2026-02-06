import pandas as pd


def aggregate_results(results):
    df = pd.DataFrame(results)

    summary = (
        df.groupby(["representation", "model", "split"])
        .agg({"rmse": ["mean", "std"], "r2": ["mean", "std"]})
        .reset_index()
    )

    summary.columns = [
        "representation",
        "model",
        "split",
        "rmse_mean",
        "rmse_std",
        "r2_mean",
        "r2_std",
    ]

    return summary
