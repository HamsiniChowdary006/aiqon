"""Quick exploratory data analysis -> results/figures/eda_overview.png + results/eda_summary.txt

Run:  python src/eda.py
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
THRESH = 12.0

df = pd.read_csv(ROOT / "data" / "port_congestion_400.csv", parse_dates=["arrival_time"])
df["util"] = df["queue_length"] / df["berth_capacity"]
df["bad_weather"] = ((df["wind_speed_kmh"] > 30) | (df["rain_mm_hr"] > 6)).astype(int)
df["is_night"] = ((df["hour"] >= 20) | (df["hour"] < 6)).astype(int)

fig, ax = plt.subplots(2, 2, figsize=(11, 8))

# 1. delay distribution
a = ax[0, 0]
a.hist(df["delay_hours"], bins=30, color="#4C72B0", edgecolor="white")
a.axvline(THRESH, color="red", ls="--", label=f"Congestion threshold = {THRESH:.0f} h")
a.set_xlabel("Waiting time before berthing (hours)"); a.set_ylabel("Vessels")
a.set_title(f"Delay distribution ({df.congested.mean():.0%} congested)"); a.legend()

# 2. per-port congestion rate
a = ax[0, 1]
port = df.groupby("port").agg(rate=("congested", "mean"), berths=("berth_capacity", "first"),
                              n=("congested", "size")).sort_values("berths")
bars = a.bar(port.index, port["rate"], color="#55A868")
for b, (_, r) in zip(bars, port.iterrows()):
    a.text(b.get_x() + b.get_width() / 2, r["rate"] + 0.01, f"{int(r.berths)} berths\nn={int(r.n)}",
           ha="center", fontsize=8)
a.set_ylim(0, port["rate"].max() + 0.15); a.set_ylabel("Congestion rate")
a.set_title("Congestion rate by port (smaller ports congest more)")
a.tick_params(axis="x", labelsize=8)

# 3. weather x queue interaction heatmap
a = ax[1, 0]
df["util_band"] = pd.cut(df["util"], [-0.01, 0.5, 1.0, 99], labels=["<0.5", "0.5-1.0", ">1.0"])
heat = df.pivot_table(index="util_band", columns="bad_weather", values="congested", aggfunc="mean", observed=False)
cnt = df.pivot_table(index="util_band", columns="bad_weather", values="congested", aggfunc="size", observed=False)
im = a.imshow(heat.values, cmap="Reds", vmin=0, vmax=1, aspect="auto")
for i in range(heat.shape[0]):
    for j in range(heat.shape[1]):
        v, n = heat.values[i, j], cnt.values[i, j]
        a.text(j, i, "n/a" if np.isnan(v) else f"{v:.0%}\n(n={int(n)})", ha="center", va="center", fontsize=10,
               color="white" if (not np.isnan(v) and v > 0.6) else "black")
a.set_xticks([0, 1]); a.set_xticklabels(["Normal weather", "Bad weather"])
a.set_yticks(range(3)); a.set_yticklabels(heat.index)
a.set_ylabel("Queue / berths"); a.set_title("Congestion rate: queue pressure x weather")
plt.colorbar(im, ax=a, fraction=0.046)

# 4. seasonality
a = ax[1, 1]
m = df.groupby(df["arrival_time"].dt.month)["congested"].mean()
colors = ["#C44E52" if k in (6, 7, 8, 9) else "#8C8C8C" for k in m.index]
a.bar(m.index, m.values, color=colors)
a.set_xticks(range(1, 13)); a.set_xlabel("Month"); a.set_ylabel("Congestion rate")
a.set_title("Seasonality (red = monsoon months Jun-Sep)")

plt.tight_layout()
(ROOT / "results" / "figures").mkdir(parents=True, exist_ok=True)
plt.savefig(ROOT / "results" / "figures" / "eda_overview.png", dpi=140)

# ---- text summary
night = df.groupby("is_night")["congested"].mean()
wkend = df.groupby(df["arrival_time"].dt.dayofweek >= 5)["congested"].mean()
lines = [
    f"Rows: {len(df)} | period {df.arrival_time.min().date()} to {df.arrival_time.max().date()}",
    f"Congestion rate overall: {df.congested.mean():.1%} (class balance target 20-40%)",
    f"Delay hours: mean {df.delay_hours.mean():.1f}, median {df.delay_hours.median():.1f}, max {df.delay_hours.max():.1f}",
    "Congestion rate by port:\n" + port[["berths", "n", "rate"]].round(2).to_string(),
    f"Night vs day congestion: {night.get(1, np.nan):.1%} vs {night.get(0, np.nan):.1%}",
    f"Weekend vs weekday congestion: {wkend.get(True, np.nan):.1%} vs {wkend.get(False, np.nan):.1%}",
    "Queue pressure x weather (congestion rate):\n" + heat.round(2).to_string(),
    "Cell sizes:\n" + cnt.to_string(),
    "Missing values:\n" + df.isna().sum()[df.isna().sum() > 0].to_string(),
]
text = "\n\n".join(lines)
(ROOT / "results" / "eda_summary.txt").write_text(text, encoding="utf-8")
print(text)
