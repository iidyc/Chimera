"""
Visualize estimated vs true distances from dist_decomp.csv.
Usage: python plot_dist_decomp.py [path/to/dist_decomp.csv]
"""
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

csv_path = sys.argv[1] if len(sys.argv) > 1 else "dist_decomp.csv"
df = pd.read_csv(csv_path)

# Ratio of estimated distance to true distance
df["ratio"] = df["est_dist"] / df["true_dist"]
ratios = df["ratio"].dropna()
ratios = ratios[np.isfinite(ratios)]
mean_r = ratios.mean()
median_r = ratios.median()

print(np.corrcoef(df["est_dist"], df["true_dist"])[0, 1])

# ---------- style ----------
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 12,
    "axes.labelsize": 13,
    "axes.titlesize": 15,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "axes.spines.top": False,
})

fig, ax = plt.subplots(figsize=(16, 6))

if "rank" in df.columns:
    rank_col = "rank"
else:
    df["rank"] = df.groupby("query_id")["true_dist"].rank(ascending=False, method="first").astype(int)
    rank_col = "rank"

# Bin ranks into buckets
max_rank = int(df[rank_col].max())
n_bins = 50
bin_edges = np.linspace(0.5, max_rank + 0.5, n_bins + 1)
df["rank_bin"] = pd.cut(df[rank_col], bins=bin_edges, labels=False)
bin_size = int(round((max_rank) / n_bins))

bin_stats = df.groupby("rank_bin").agg(
    mean_true=("true_dist", "mean"),
    mean_est=("est_dist", "mean"),
    ratio_mean=("ratio", "mean"),
    ratio_std=("ratio", "std"),
).reindex(range(n_bins))

bar_width = 0.85
x = np.arange(n_bins)

# --- Stacked bars ---
bars_est = ax.bar(x, bin_stats["mean_est"], bar_width,
                  label="Estimated distance (partial)", color="#4C8BF5", edgecolor="white", linewidth=0.3)
residual = (bin_stats["mean_true"] - bin_stats["mean_est"]).clip(lower=0)
bars_res = ax.bar(x, residual, bar_width, bottom=bin_stats["mean_est"],
                  label="Residual (true − estimated)", color="#F57C4C", alpha=0.85, edgecolor="white", linewidth=0.3)

ax.set_ylabel("Mean Distance Score")
ax.set_xlabel(f"True Distance Rank Bucket  (each bucket ≈ {bin_size} docs)")

# X-ticks: show every ~5th bucket, using midpoint rank
step = max(1, n_bins // 12)
tick_positions = list(range(0, n_bins, step))
tick_labels = [f"{int(bin_edges[i]+0.5)}–{int(bin_edges[i+1]-0.5)}" for i in tick_positions]
ax.set_xticks(tick_positions)
ax.set_xticklabels(tick_labels, rotation=35, ha="right")

# Light horizontal grid on bar axis
ax.yaxis.set_major_locator(mticker.MaxNLocator(6))
ax.grid(axis="y", linestyle=":", alpha=0.4)
ax.set_axisbelow(True)

# --- Ratio on secondary y-axis ---
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

# --- Combined legend ---
handles1, labels1 = ax.get_legend_handles_labels()
handles2, labels2 = ax2.get_legend_handles_labels()
ax.legend(handles1 + handles2, labels1 + labels2,
          loc="upper right", framealpha=0.9, edgecolor="lightgray")

ax.set_title("Estimated vs True Distance by Rank  (with ratio ± std)", pad=12)

plt.tight_layout()
plt.savefig("dist_decomp.png", dpi=200, bbox_inches="tight")
plt.savefig("dist_decomp.pdf", bbox_inches="tight")
print(f"Saved dist_decomp.png and dist_decomp.pdf")
print(f"\nSummary across {len(df)} (query, doc) pairs from {df['query_id'].nunique()} queries:")
print(f"  Ratio (est/true) — mean: {mean_r:.4f}, median: {median_r:.4f}, "
      f"std: {ratios.std():.4f}, min: {ratios.min():.4f}, max: {ratios.max():.4f}")
