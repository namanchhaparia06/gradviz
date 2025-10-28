from typing import Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def plot_lines_by_key(df: pd.DataFrame, by: str = "layer", topk: int = 20, figsize=(10,6)):
    assert by in ("layer", "param")
    key = by
    # aggregate by key and take topk by last-step mean norm
    last_step = df["step"].max()
    agg = df[df["step"] == last_step].groupby(key)["grad_norm"].mean().sort_values(ascending=False)
    top = agg.head(topk).index.tolist()
    use = df[df[key].isin(top)]

    plt.figure(figsize=figsize)
    for name, g in use.groupby(key):
        g_sorted = g.sort_values("step")
        plt.plot(g_sorted["step"], g_sorted["grad_norm"], label=name)
    plt.xlabel("Step")
    plt.ylabel("Grad L2 norm")
    plt.title(f"Gradient norms over steps by {by} (top {topk})")
    plt.legend(loc="best", fontsize=8, ncol=2)
    return plt.gca()

def plot_heatmap_at_step(df: pd.DataFrame, by: str = "layer", at_step: Optional[int] = None, figsize=(10,6)):
    assert by in ("layer", "param")
    if at_step is None:
        at_step = df["step"].max()
    # take nearest step if exact not present
    steps = df["step"].unique()
    nearest = steps[np.argmin(np.abs(steps - at_step))]
    snap = df[df["step"] == nearest]

    # sort by mean norm
    order = snap.groupby(by)["grad_norm"].mean().sort_values(ascending=False).index.tolist()
    values = snap.groupby(by)["grad_norm"].mean().reindex(order).values

    plt.figure(figsize=figsize)
    plt.imshow(values[None, :], aspect="auto")
    plt.yticks([])
    plt.xticks(range(len(order)), order, rotation=90, fontsize=8)
    plt.colorbar(label="Grad L2 norm")
    plt.title(f"Gradient norms heatmap by {by} @ step {nearest}")
    return plt.gca()
