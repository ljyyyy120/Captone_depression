import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

def spearman_corr_heatmap(df, columns=None, figsize=(8, 6), savepath=None):
    """
    Spearman pairwise correlation heatmap (lower triangle only),
    show white grid between cells, only mark * for p<0.05,
    all fonts Arial 10pt, no frame.
    """
    plt.rcParams['font.family'] = 'Arial'
    plt.rcParams['font.size'] = 10

    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()

    cols = list(columns)
    n = len(cols)
    corr = pd.DataFrame(np.eye(n), index=cols, columns=cols, dtype=float)
    pvals = pd.DataFrame(np.zeros((n, n)), index=cols, columns=cols, dtype=float)

    # compute Spearman
    for i in range(n):
        for j in range(i, n):
            x, y = df[cols[i]], df[cols[j]]
            valid = x.notna() & y.notna()
            if valid.sum() >= 3:
                r, p = spearmanr(x[valid], y[valid])
            else:
                r, p = np.nan, np.nan
            corr.iloc[i, j] = corr.iloc[j, i] = r
            pvals.iloc[i, j] = pvals.iloc[j, i] = p

    # mask upper triangle
    mask = np.triu_indices_from(corr, k=0)
    corr_masked = corr.copy()
    corr_masked.values[mask] = np.nan

    fig, ax = plt.subplots(figsize=figsize)

    # plot
    cmap = plt.get_cmap("coolwarm")
    im = ax.pcolormesh(corr_masked.values, cmap=cmap, vmin=-1, vmax=1,
                       edgecolors="white", linewidth=0.5)

    # axis labels
    ax.set_xticks(np.arange(n) + 0.5, labels=cols, rotation=45, ha="right")
    ax.set_yticks(np.arange(n) + 0.5, labels=cols)


    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.invert_yaxis()

    # annotation
    for i in range(n):
        for j in range(i):
            p = pvals.iloc[i, j]
            if not np.isnan(p) and p < 0.05:
                ax.text(j + 0.5, i + 0.5, "*", ha="center", va="center",
                        fontsize=10, color="black", fontweight="bold")

    # colorbar
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Spearman ρ", rotation=90, fontsize=10)

    ax.set_title("Pairwise Spearman Correlation", pad=15, fontsize=10)
    fig.tight_layout()

    if savepath:
        fig.savefig(savepath, dpi=600, bbox_inches="tight")

    return corr, pvals