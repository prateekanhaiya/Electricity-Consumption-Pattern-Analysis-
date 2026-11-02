"""Generate a realistic synthetic hourly electricity-consumption dataset.

Use this when you don't have a real dataset. The output CSV has the columns:
    timestamp, consumption_kwh, temperature_c
"""
from pathlib import Path

import numpy as np
import pandas as pd


def generate(days: int = 365, seed: int = 42,
             out: str = "data/electricity_consumption.csv") -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2023-01-01", periods=days * 24, freq="h")
    hour = idx.hour.values
    doy = idx.dayofyear.values
    dow = idx.dayofweek.values

    # Temperature: yearly cycle + daily cycle + noise
    temp = (25 + 10 * np.sin(2 * np.pi * (doy - 110) / 365)
            + 4 * np.sin(2 * np.pi * (hour - 9) / 24)
            + rng.normal(0, 1.5, len(idx)))

    # Daily load shape: base + morning peak + evening peak
    base = 0.4
    morning = 0.9 * np.exp(-((hour - 7.5) ** 2) / (2 * 1.5 ** 2))
    evening = 1.6 * np.exp(-((hour - 20) ** 2) / (2 * 2.0 ** 2))
    weekend_factor = np.where(dow >= 5, 1.15, 1.0)
    cooling = 0.08 * np.clip(temp - 26, 0, None)        # AC usage on hot hours
    heating = 0.05 * np.clip(15 - temp, 0, None)

    cons = (base + morning + evening) * weekend_factor + cooling + heating
    cons += rng.normal(0, 0.08, len(idx))

    # Inject anomalies (spikes) so the detector has something to find
    spikes = rng.choice(len(idx), size=30, replace=False)
    cons[spikes] *= rng.uniform(2.5, 3.5, size=30)
    cons = np.clip(cons, 0.05, None)

    df = pd.DataFrame({"timestamp": idx,
                       "consumption_kwh": cons.round(3),
                       "temperature_c": temp.round(2)})

    # Inject missing values (~0.5%) to demonstrate cleaning
    miss = rng.choice(len(df), size=int(0.005 * len(df)), replace=False)
    df.loc[miss, "consumption_kwh"] = np.nan

    Path(out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    return df


if __name__ == "__main__":
    d = generate()
    print(f"Generated {len(d)} rows -> data/electricity_consumption.csv")
