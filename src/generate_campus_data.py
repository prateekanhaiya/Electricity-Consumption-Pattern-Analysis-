"""Synthetic campus electricity readings for the DA-018 Energy Analytics notebook.

Meter data layout (long format), one row per area per 15-minute interval:
    timestamp, area, consumption_kwh, temperature_c

Units: consumption_kwh = energy used during that 15-minute interval (kWh).
       Average power in kW = consumption_kwh * 4.

Deliberately included data problems (so the cleaning logic has real work to do):
    - missing intervals (random single rows, a 3 h gap, one full-day outage)
    - a few missing values, duplicate rows, negative (invalid) readings
    - isolated spikes and "equipment left on" weekend events in Computer Labs
"""
from pathlib import Path

import numpy as np
import pandas as pd


def _window(hour, start, end):
    """Smooth 0->1->0 occupancy window between start and end hour."""
    return 1 / (1 + np.exp(-(hour - start) * 3)) * 1 / (1 + np.exp((hour - end) * 3))


def generate_campus(seed: int = 7,
                    out: str = "data/campus_energy_readings.csv") -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2023-01-01", "2023-12-31 23:45", freq="15min")
    hour = (idx.hour + idx.minute / 60).values
    dow = idx.dayofweek.values
    doy = idx.dayofyear.values
    month = idx.month.values
    weekday = (dow < 5).astype(float)

    temp = (26 + 9 * np.sin(2 * np.pi * (doy - 110) / 365)
            + 4 * np.sin(2 * np.pi * (hour - 9) / 24)
            + rng.normal(0, 1.2, len(idx)))
    cool = np.clip(temp - 26, 0, None)

    # Academic calendar: reduced occupancy in May and June (summer break)
    vac = np.select([month == 6, month == 5], [0.5, 0.8], default=1.0)

    kw = {}
    kw["Academic Block"] = (8 + (6 + 39 * weekday) * _window(hour, 8, 17) * vac
                            + 1.0 * cool * _window(hour, 8, 17) * weekday)
    kw["Library"] = (6 + np.where(dow < 5, 28, 18) * _window(hour, 9, 21)
                     * np.where(vac < 1, 0.7, 1.0)
                     + 0.8 * cool * _window(hour, 9, 21))
    hostel_vac = np.select([month == 6, month == 5], [0.55, 0.85], default=1.0)
    kw["Hostel"] = ((15 + 30 * _window(hour, 18, 23.5) + 14 * _window(hour, 5.5, 8.5))
                    * np.where(dow >= 5, 1.1, 1.0) * hostel_vac
                    + 1.5 * cool * (0.5 + 0.5 * _window(hour, 18, 24)))
    labs = (12 + (np.where(dow < 5, 55, 5)) * _window(hour, 9, 18) * vac
            + 0.5 * cool * _window(hour, 9, 18))
    # "Left on" events: Friday evening -> Monday morning on four weekends
    for start in ["2023-02-10 19:00", "2023-05-05 19:00",
                  "2023-08-18 19:00", "2023-11-03 19:00"]:
        s = pd.Timestamp(start)
        e = s + pd.Timedelta(hours=59)  # until Monday 06:00
        labs = labs + 28 * ((idx >= s) & (idx < e))
    kw["Computer Labs"] = labs

    frames = []
    for area, power in kw.items():
        kwh = power * 0.25 * (1 + rng.normal(0, 0.05, len(idx)))
        # isolated spikes
        sp = rng.choice(len(idx), size=8, replace=False)
        kwh[sp] *= rng.uniform(2.5, 3.5, size=8)
        frames.append(pd.DataFrame({"timestamp": idx, "area": area,
                                    "consumption_kwh": kwh.round(3),
                                    "temperature_c": temp.round(2)}))
    df = pd.concat(frames, ignore_index=True)

    # ---- inject data-quality problems --------------------------------
    # invalid negative readings
    neg = rng.choice(len(df), size=6, replace=False)
    df.loc[neg, "consumption_kwh"] = -df.loc[neg, "consumption_kwh"].abs()
    # missing values (NaN)
    nan_i = rng.choice(len(df), size=int(0.001 * len(df)), replace=False)
    df.loc[nan_i, "consumption_kwh"] = np.nan
    # missing intervals (rows absent): random singles
    drop = rng.choice(len(df), size=int(0.003 * len(df)), replace=False)
    df = df.drop(index=drop)
    # 3-hour gap in Library, full-day outage in Hostel
    gap = ((df.area == "Library")
           & (df.timestamp >= "2023-03-21 10:00") & (df.timestamp < "2023-03-21 13:00"))
    outage = (df.area == "Hostel") & (df.timestamp.dt.date == pd.Timestamp("2023-08-14").date())
    df = df[~(gap | outage)]
    # duplicate rows
    df = pd.concat([df, df.sample(10, random_state=seed)], ignore_index=True)

    df = df.sort_values(["timestamp", "area"]).reset_index(drop=True)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    return df


if __name__ == "__main__":
    d = generate_campus()
    print(f"Wrote {len(d):,} rows -> data/campus_energy_readings.csv")
