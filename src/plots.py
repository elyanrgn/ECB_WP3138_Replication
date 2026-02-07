import pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

with open("models\\tcvar_ar1_results.pkl", "rb") as f:
    res = pickle.load(f)

trends = res["trends"]
dates = res["dates"]
data_values = res["data_values"]

nber_recessions = [
    ("1960-04-01", "1961-02-01"),
    ("1970-01-01", "1970-11-01"),
    ("1973-11-01", "1975-03-01"),
    ("1980-02-01", "1980-07-01"),
    ("1981-08-01", "1982-11-01"),
    ("1990-08-01", "1991-03-01"),
    ("2001-04-01", "2001-11-01"),
    ("2008-01-01", "2009-06-01"),
    ("2020-03-01", "2020-04-01"),
]  # Source : FRED


def shade_nber(ax):
    for start, end in nber_recessions:
        ax.axvspan(
            pd.Timestamp(start), pd.Timestamp(end), color="gray", alpha=0.15, zorder=0
        )


trends = np.array(res["trends"])
cycles = np.array(res["cycles"])

# TRENDS
g_star_med = np.median(trends[:, :, 0], axis=0)
g_star_low = np.percentile(trends[:, :, 0], 16, axis=0)
g_star_high = np.percentile(trends[:, :, 0], 84, axis=0)

pi_star_med = np.median(trends[:, :, 1], axis=0)
pi_star_low = np.percentile(trends[:, :, 1], 16, axis=0)
pi_star_high = np.percentile(trends[:, :, 1], 84, axis=0)

z_star_med = np.median(trends[:, :, 2], axis=0)  # Pref

# REAL NATURAL RATE (r* = z* + g*)
r_star_draws = trends[:, :, 2] + trends[:, :, 0]
r_star_med = np.median(r_star_draws, axis=0)
r_star_low = np.percentile(r_star_draws, 16, axis=0)
r_star_high = np.percentile(r_star_draws, 84, axis=0)

# NOMINAL NATURAL RATE (i* = r* + pi*)
i_star_draws = r_star_draws + trends[:, :, 1]
i_star_med = np.median(i_star_draws, axis=0)
i_star_low = np.percentile(i_star_draws, 16, axis=0)
i_star_high = np.percentile(i_star_draws, 84, axis=0)

# CYCLES (GAPS)
# Output Gap
output_gap_med = np.median(cycles[:, :, 0], axis=0)
output_gap_low = np.percentile(cycles[:, :, 0], 16, axis=0)
output_gap_high = np.percentile(cycles[:, :, 0], 84, axis=0)

# Inflation Gap
inflation_gap_med = np.median(cycles[:, :, 1], axis=0)
inflation_gap_low = np.percentile(cycles[:, :, 1], 16, axis=0)
inflation_gap_high = np.percentile(cycles[:, :, 1], 84, axis=0)

# Interest Rate Gap
i_star_gap_med = np.median(cycles[:, :, 2], axis=0)
i_star_gap_low = np.percentile(cycles[:, :, 2], 16, axis=0)
i_star_gap_high = np.percentile(cycles[:, :, 2], 84, axis=0)

# Commodity Gap
commodity_gap_med = np.median(cycles[:, :, 4], axis=0)  # Index 4 pour Commo Cycle
commodity_gap_low = np.percentile(cycles[:, :, 4], 16, axis=0)
commodity_gap_high = np.percentile(cycles[:, :, 4], 84, axis=0)


# Data, Median Estimates, 68% Coverage Bands
fig, axes = plt.subplots(3, 1, figsize=(6, 10))

# Panel 1: GDP Growth
axes[0].plot(dates, data_values[:, 0], "k--", alpha=0.6, linewidth=1.5, label="GDP")
axes[0].plot(dates, g_star_med, "b-", linewidth=2, label="$\Delta g^*$")
axes[0].fill_between(dates, g_star_low, g_star_high, color="blue", alpha=0.2)
axes[0].set_title("Gross domestic product", fontsize=12)
axes[0].set_ylabel("percentage change")
axes[0].legend(loc="upper right")
axes[0].grid(True, alpha=0.3)
axes[0].set_ylim(-6, 12)
shade_nber(axes[0])

# Panel 2: Inflation
axes[1].plot(dates, data_values[:, 1], "k--", alpha=0.6, linewidth=1.5, label="$\pi$")
axes[1].plot(dates, pi_star_med, "b-", linewidth=2, label="$\pi^*$")
axes[1].fill_between(dates, pi_star_low, pi_star_high, color="blue", alpha=0.2)
# Proxy inflation expectations = Trend + part of cycle (simple diff here)
axes[1].plot(
    dates,
    data_values[:, 1] - inflation_gap_med,
    "r-",
    linewidth=1.5,
    alpha=0.7,
    label="$\pi^e$ (proxy)",
)
axes[1].set_title("Inflation", fontsize=12)
axes[1].set_ylabel("percentage change")
axes[1].legend(loc="upper right")
axes[1].grid(True, alpha=0.3)
axes[1].set_ylim(-4, 14)
shade_nber(axes[1])

# Panel 3: Nominal Interest Rate
mask_rate = ~np.isnan(data_values[:, 2])
axes[2].plot(
    dates[mask_rate],
    data_values[mask_rate, 2],
    "k--",
    alpha=0.6,
    linewidth=1.5,
    label="$i$",
)
axes[2].plot(dates, i_star_med, "b-", linewidth=2, label="$i^*$")
axes[2].fill_between(dates, i_star_low, i_star_high, color="blue", alpha=0.2)
axes[2].set_title("3-month nominal treasury yield", fontsize=12)
axes[2].set_ylabel("percent")
axes[2].legend(loc="upper right")
axes[2].grid(True, alpha=0.3)
axes[2].set_ylim(-2, 14)
shade_nber(axes[2])

plt.tight_layout()
plt.savefig("outputs\\figure2_replication_final.png", dpi=300)
plt.show()


# Real Natural Rate of Interest (r*)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

ax1.plot(dates, r_star_med, "b-", linewidth=2, label="$r^*$")
ax1.fill_between(dates, r_star_low, r_star_high, color="blue", alpha=0.2)
ax1.axhline(0, color="k", linestyle="--", linewidth=0.5)
ax1.set_title("real natural rate of interest", fontsize=12)
ax1.set_ylabel("percent")
ax1.legend()
ax1.grid(True, alpha=0.3)
ax1.set_ylim(-6, 12)
shade_nber(ax1)

# Right: Changes in trend
try:
    idx_1996 = np.where(dates.year == 1996)[0][0]
    idx_2025 = np.where(dates.year == 2025)[0][0]

    change_r_star = r_star_med[idx_2025] - r_star_med[idx_1996]
    change_g_star = g_star_med[idx_2025] - g_star_med[idx_1996]
    change_z_star = z_star_med[idx_2025] - z_star_med[idx_1996]
    # Simple Bootstrap for CI on changes
    idx_sample = np.random.choice(len(trends), 1000)
    boot_r = (
        trends[idx_sample, idx_2025, 2]
        + trends[idx_sample, idx_2025, 0]
        - (trends[idx_sample, idx_1996, 2] + trends[idx_sample, idx_1996, 0])
    )
    boot_g = trends[idx_sample, idx_2025, 0] - trends[idx_sample, idx_1996, 0]
    boot_z = trends[idx_sample, idx_2025, 2] - trends[idx_sample, idx_1996, 2]

    ci_r = np.percentile(boot_r, [16, 84])
    ci_g = np.percentile(boot_g, [16, 84])
    ci_z = np.percentile(boot_z, [16, 84])

    ax2.axis("off")
    table_text = f"""
    Changes in Trend, 1996-2023
    68% coverage bands in parentheses

    $r_t^*$                   {change_r_star:.2f}
                              ({ci_r[0]:.2f}, {ci_r[1]:.2f})

    $\Delta g_t^*$            {change_g_star:.2f}
                              ({ci_g[0]:.2f}, {ci_g[1]:.2f})

    $z_t^*$                   {change_z_star:.2f}
                              ({ci_z[0]:.2f}, {ci_z[1]:.2f})
    """
    ax2.text(
        0.1,
        0.5,
        table_text,
        fontsize=12,
        family="monospace",
        verticalalignment="center",
    )
except IndexError:
    ax2.text(0.5, 0.5, "Date range error for Table", ha="center")

plt.tight_layout()
plt.savefig("outputs\\figure3_replication_final.png", dpi=300)
plt.show()


# Gaps
exp_gap_med = np.median(cycles[:, :, 3], axis=0)
exp_gap_low = np.percentile(cycles[:, :, 3], 16, axis=0)
exp_gap_high = np.percentile(cycles[:, :, 3], 84, axis=0)

fig, axes = plt.subplots(2, 3, figsize=(14, 8))
axes = axes.flatten()

# 1. Output Gap
axes[0].plot(dates, output_gap_med, "b-", linewidth=1.5)
axes[0].fill_between(dates, output_gap_low, output_gap_high, color="blue", alpha=0.2)
axes[0].axhline(0, color="k", linestyle="--")
axes[0].set_title("Output gap")
axes[0].set_ylim(-10, 10)
shade_nber(axes[0])

# 2. Inflation Gap
axes[1].plot(dates, inflation_gap_med, "b-", linewidth=1.5)
axes[1].fill_between(
    dates, inflation_gap_low, inflation_gap_high, color="blue", alpha=0.2
)
axes[1].axhline(0, color="k", linestyle="--")
axes[1].set_title("Inflation gap")
axes[1].set_ylim(-10, 10)
shade_nber(axes[1])


# 4. Interest Rate Gap
axes[3].plot(dates, i_star_gap_med, "b-", linewidth=1.5)
axes[3].fill_between(dates, i_star_gap_low, i_star_gap_high, color="blue", alpha=0.2)
axes[3].axhline(0, color="k", linestyle="--")
axes[3].set_title("Nominal interest rate gap")
axes[3].set_ylim(-10, 10)
shade_nber(axes[3])

axes[2].axis("off")
axes[4].axis("off")
axes[5].axis("off")
plt.tight_layout()
plt.savefig("outputs\\figure4_replication_final.png", dpi=300)
plt.show()
