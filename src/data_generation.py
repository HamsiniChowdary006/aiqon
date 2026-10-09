"""Synthetic port-congestion dataset (400 vessel arrivals, 4 ports, 12 months).

Ground truth
------------
delay_hours  = waiting time before berthing (hours), simulated by a formula + noise
congested    = 1 if delay_hours > CONGESTION_THRESHOLD_H (12h), else 0

Simulation rules (these are OUR assumptions, not real-world facts):
  1. Utilisation pressure: queue_length / berth_capacity (nonlinear above 1.0)
  2. Incoming vessels in next 24h, scaled by berth capacity
  3. Bad weather (wind, rain) adds delay, and amplifies the queue effect (interaction)
  4. Night and weekend arrivals wait longer (reduced labour / pilots)
  5. Monsoon months (Jun-Sep) add delay
  6. Bigger vessels need more berth time
  7. Each port has its own base delay (efficiency)
  8. Gaussian-ish noise so the label is not deterministic

Real-world messiness added on purpose:
  - noisy sensor readings on wind speed
  - ~5% missing values in wind, rain, queue_length
  - recent_avg_delay_hrs uses ONLY earlier vessels at the same port (no leakage)

Run:  python src/data_generation.py
"""
import numpy as np
import pandas as pd

SEED = 42
N_ROWS = 400
CONGESTION_THRESHOLD_H = 12.0

# port: (berth_capacity, base_delay_hours, arrival_weight)
PORTS = {
    "Mundra":       (10, 1.0, 0.32),
    "Nhava Sheva":  (8,  2.0, 0.30),
    "Chennai":      (5,  2.5, 0.20),
    "Visakhapatnam": (4, 3.0, 0.18),
}


def generate(n=N_ROWS, seed=SEED, missing_rate=0.05, wind_noise_sd=4.0):
    rng = np.random.default_rng(seed)
    names = list(PORTS)
    weights = [PORTS[p][2] for p in names]

    # --- arrival times over 12 months, sorted (needed for time-based split)
    start = pd.Timestamp("2025-01-01")
    offsets = np.sort(rng.uniform(0, 365 * 24, n))
    arrival = start + pd.to_timedelta(offsets, unit="h")
    arrival = arrival.round("h")

    port = rng.choice(names, size=n, p=weights)
    cap = np.array([PORTS[p][0] for p in port])
    base = np.array([PORTS[p][1] for p in port])

    hour = arrival.hour.values
    dow = arrival.dayofweek.values
    month = arrival.month.values
    is_night = ((hour >= 20) | (hour < 6)).astype(int)
    is_weekend = (dow >= 5).astype(int)
    monsoon = np.isin(month, [6, 7, 8, 9]).astype(int)

    # --- traffic: scales with port size, heavier in some periods
    queue = rng.poisson(lam=cap * rng.uniform(0.35, 1.15, n))
    incoming = rng.poisson(lam=cap * rng.uniform(0.4, 1.3, n))

    # --- weather (true values), stronger in monsoon
    wind_true = np.clip(rng.gamma(4.0, 4.0, n) + monsoon * 8, 0, 80)    # km/h
    rain = np.clip(rng.exponential(1.5, n) * (1 + 2 * monsoon) - 1, 0, None)  # mm/hr
    bad_weather = ((wind_true > 30) | (rain > 6)).astype(int)

    vessel_teu = np.clip(rng.normal(6000, 3000, n), 500, 15000)         # vessel size

    # --- delay formula
    util = queue / cap
    util_effect = 2.2 * util + 9.0 * np.clip(util - 1.0, 0, None) ** 1.5   # jump above capacity
    incoming_effect = 2.5 * (incoming / cap)
    weather_effect = 2.5 * bad_weather + 0.04 * wind_true + 0.25 * rain
    interaction = 4.0 * util * bad_weather                                  # queue x weather
    time_effect = 1.5 * is_night + 1.2 * is_weekend + 1.5 * monsoon
    size_effect = vessel_teu / 5000.0
    noise = rng.normal(0, 2.0, n)

    delay = base + util_effect + incoming_effect + weather_effect + interaction \
        + time_effect + size_effect + noise
    delay = np.clip(delay, 0, None)

    # --- leakage-free rolling history: mean delay of the previous 5 vessels AT THE SAME PORT
    recent = np.full(n, np.nan)
    history = {p: [] for p in names}
    for i in range(n):  # rows are already time-sorted
        h = history[port[i]]
        if h:
            recent[i] = np.mean(h[-5:])
        h.append(delay[i])   # appended AFTER use -> current delay never leaks

    df = pd.DataFrame({
        "arrival_time": arrival,
        "port": port,
        "hour": hour,
        "day_of_week": dow,
        "month": month,
        "berth_capacity": cap,
        "queue_length": queue.astype(float),
        "incoming_24h": incoming,
        "wind_speed_kmh": wind_true + rng.normal(0, wind_noise_sd, n),   # noisy sensor
        "rain_mm_hr": rain,
        "vessel_size_teu": vessel_teu.round(0),
        "recent_avg_delay_hrs": recent,
        "delay_hours": delay.round(2),
    })
    df["congested"] = (df["delay_hours"] > CONGESTION_THRESHOLD_H).astype(int)

    # --- missing values
    for col in ["wind_speed_kmh", "rain_mm_hr", "queue_length"]:
        df.loc[rng.random(n) < missing_rate, col] = np.nan
    df["wind_speed_kmh"] = df["wind_speed_kmh"].clip(lower=0)
    return df


if __name__ == "__main__":
    df = generate()
    df.to_csv("data/port_congestion_400.csv", index=False)
    print(df.shape)
    print("Congestion rate:", round(df.congested.mean(), 3))
    print(df.groupby("port").agg(n=("congested", "size"), rate=("congested", "mean"),
                                 mean_delay=("delay_hours", "mean")).round(2))
    print("Missing:\n", df.isna().sum()[df.isna().sum() > 0])
    print(df.delay_hours.describe().round(2))
