"""
Plotting utilities for Chimera experiments.

Usage:
  python plot.py [dist_decomp.csv] [gather_recall.csv]

Both CSV paths are optional and default to dist_decomp.csv / gather_recall.csv.
Produces a single figure with two side-by-side panels.
"""
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# ---------- global style ----------
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "axes.spines.top": False,
})


# ============================================================
# Panel 1 — Distance decomposition (estimated vs true by rank)
# ============================================================
def plot_dist_decomp(ax, csv_path="dist_decomp.csv"):
    df = pd.read_csv(csv_path)
    df["ratio"] = df["est_dist"] / df["true_dist"]
    ratios = df["ratio"].dropna()
    ratios = ratios[np.isfinite(ratios)]
    mean_r = ratios.mean()
    median_r = ratios.median()

    if "rank" in df.columns:
        rank_col = "rank"
    else:
        df["rank"] = df.groupby("query_id")["true_dist"].rank(ascending=False, method="first").astype(int)
        rank_col = "rank"

    max_rank = int(df[rank_col].max())
    n_bins = 50
    bin_edges = np.linspace(0.5, max_rank + 0.5, n_bins + 1)
    df["rank_bin"] = pd.cut(df[rank_col], bins=bin_edges, labels=False)
    bin_size = int(round(max_rank / n_bins))

    bin_stats = df.groupby("rank_bin").agg(
        mean_true=("true_dist", "mean"),
        mean_est=("est_dist", "mean"),
        ratio_mean=("ratio", "mean"),
        ratio_std=("ratio", "std"),
    ).reindex(range(n_bins))

    bar_width = 0.85
    x = np.arange(n_bins)

    ax.bar(x, bin_stats["mean_est"], bar_width,
           label="Estimated distance (partial)", color="#4C8BF5", edgecolor="white", linewidth=0.3)
    residual = (bin_stats["mean_true"] - bin_stats["mean_est"]).clip(lower=0)
    ax.bar(x, residual, bar_width, bottom=bin_stats["mean_est"],
           label="Residual (true − estimated)", color="#F57C4C", alpha=0.85, edgecolor="white", linewidth=0.3)

    ax.set_ylabel("Mean Distance Score")
    ax.set_xlabel(f"True Distance Rank Bucket  (≈ {bin_size} docs each)")

    step = max(1, n_bins // 10)
    tick_positions = list(range(0, n_bins, step))
    tick_labels = [f"{int(bin_edges[i]+0.5)}–{int(bin_edges[i+1]-0.5)}" for i in tick_positions]
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, rotation=35, ha="right")
    ax.yaxis.set_major_locator(mticker.MaxNLocator(6))
    ax.grid(axis="y", linestyle=":", alpha=0.4)
    ax.set_axisbelow(True)

    ax2 = ax.twinx()
    ratio_mean = bin_stats["ratio_mean"]
    ratio_std = bin_stats["ratio_std"]
    ax2.fill_between(x, ratio_mean - ratio_std, ratio_mean + ratio_std,
                     alpha=0.24, color="black", label="Ratio ± 1 std")
    ax2.plot(x, ratio_mean, color="black", linewidth=2, marker="o", markersize=3.5,
             markerfacecolor="white", markeredgewidth=1.2, label="Mean ratio (est / true)")
    ax2.axhline(1.0, color="gray", linestyle="--", linewidth=0.7, alpha=0.5)
    ax2.set_ylabel("Estimated / True Ratio")
    y_hi = max(1.15, float((ratio_mean + ratio_std).max()) * 1.08)
    ax2.set_ylim(0, y_hi)
    ax2.spines["top"].set_visible(False)

    handles1, labels1 = ax.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(handles1 + handles2, labels1 + labels2,
              loc="upper right", framealpha=0.9, edgecolor="lightgray", fontsize=8)
    ax.set_title("Estimated vs True Distance by Rank", pad=10)

    print(f"[dist] {len(df)} pairs, {df['query_id'].nunique()} queries — "
          f"ratio mean={mean_r:.4f}, median={median_r:.4f}, std={ratios.std():.4f}")


# ============================================================
# Panel 2 — Candidate size vs Recall curve (two strategies)
# ============================================================
def plot_recall_curve(ax, csv_path="gather_recall.csv"):
    df = pd.read_csv(csv_path)

    colors = {"rank_dists": "#4C8BF5", "gather_ids": "#F57C4C"}
    markers = {"rank_dists": "o", "gather_ids": "s"}
    labels = {"rank_dists": "Rank by cluster dists (vary k)",
              "gather_ids": "Gather all IVF hits (vary nprobe)"}

    for method in df["method"].unique():
        sub = df[df["method"] == method].sort_values("avg_cand_size")
        ax.plot(sub["avg_cand_size"], sub["recall"],
                color=colors.get(method, "gray"),
                marker=markers.get(method, "^"),
                markersize=8, markeredgecolor="white", markeredgewidth=1.0,
                linewidth=2, label=labels.get(method, method))
        for _, row in sub.iterrows():
            ax.annotate(f'{int(row["param"])}',
                        (row["avg_cand_size"], row["recall"]),
                        textcoords="offset points", xytext=(6, 6),
                        fontsize=8, color=colors.get(method, "gray"), fontweight="bold")

    ax.set_xlabel("Average Candidate Set Size")
    ax.set_ylabel("Recall@100")
    ax.set_title("Candidate Generation: Size vs Recall")
    ax.grid(axis="both", linestyle=":", alpha=0.4)
    ax.set_axisbelow(True)
    ax.legend(loc="lower right", framealpha=0.9, edgecolor="lightgray")
    ax.spines["right"].set_visible(False)

    print(f"[recall] Data points:\n{df.to_string(index=False)}")


# ============================================================
# Main — single figure with two panels
# ============================================================
if __name__ == "__main__":
    dist_csv = sys.argv[1] if len(sys.argv) > 1 else "dist_decomp.csv"
    recall_csv = sys.argv[2] if len(sys.argv) > 2 else "gather_recall.csv"

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(22, 6))

    plot_dist_decomp(ax1, dist_csv)
    plot_recall_curve(ax2, recall_csv)

    plt.tight_layout()
    plt.savefig("plots.png", dpi=200, bbox_inches="tight")
    plt.savefig("plots.pdf", bbox_inches="tight")
    print(f"\nSaved plots.png and plots.pdf")
