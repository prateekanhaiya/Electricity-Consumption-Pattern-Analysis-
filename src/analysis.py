"""Core functions: load, clean, feature engineering, plots, clustering,
anomaly detection and forecasting."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # no display needed
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (mean_absolute_error, mean_squared_error,
                             r2_score, silhouette_score)

DAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


# ---------------------------------------------------------------- 1. LOAD
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    for col in ("timestamp", "consumption_kwh"):
        if col not in df.columns:
            raise ValueError(f"CSV must contain a '{col}' column")
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


# --------------------------------------------------------------- 2. CLEAN
def clean(df: pd.DataFrame):
    df = df.drop_duplicates("timestamp").sort_values("timestamp").set_index("timestamp")
    df = df.asfreq("h")                                   # enforce hourly grid
    n_missing = int(df["consumption_kwh"].isna().sum())
    df = df.interpolate(method="time", limit=6)           # fill short gaps
    df = df.dropna(subset=["consumption_kwh"])
    df = df[df["consumption_kwh"] >= 0]
    return df, n_missing


# ------------------------------------------------------------ 3. FEATURES
def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["hour"] = df.index.hour
    df["dayofweek"] = df.index.dayofweek
    df["month"] = df.index.month
    df["is_weekend"] = (df["dayofweek"] >= 5).astype(int)
    return df


# --------------------------------------------------------------- 4. PLOTS
def _save(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def make_plots(df: pd.DataFrame, fig_dir: Path) -> dict:
    fig_dir.mkdir(parents=True, exist_ok=True)
    c = "consumption_kwh"
    stats = {}

    # Hourly profile: weekday vs weekend
    prof = df.groupby(["is_weekend", "hour"])[c].mean().unstack(0)
    prof.columns = ["Weekday", "Weekend"]
    fig, ax = plt.subplots(figsize=(8, 4))
    prof.plot(ax=ax, marker="o")
    ax.set(title="Average hourly load profile", xlabel="Hour of day",
           ylabel="kWh", xticks=range(0, 24, 2))
    ax.grid(alpha=.3)
    _save(fig, fig_dir / "01_hourly_profile.png")
    stats["peak_hour"] = int(df.groupby("hour")[c].mean().idxmax())
    stats["lowest_hour"] = int(df.groupby("hour")[c].mean().idxmin())

    # Day-of-week
    dow = df.groupby("dayofweek")[c].mean()
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(DAYS, dow.values, color="#4c72b0")
    ax.set(title="Average consumption by day of week", ylabel="kWh per hour")
    _save(fig, fig_dir / "02_day_of_week.png")

    # Monthly total
    monthly = df[c].resample("MS").sum()
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(monthly.index, monthly.values, marker="o")
    ax.set(title="Monthly total consumption", ylabel="kWh")
    ax.grid(alpha=.3)
    _save(fig, fig_dir / "03_monthly_trend.png")
    stats["peak_month"] = monthly.idxmax().strftime("%B %Y")

    # Heatmap hour x weekday
    heat = df.pivot_table(values=c, index="hour", columns="dayofweek", aggfunc="mean")
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(heat.values, aspect="auto", origin="lower", cmap="YlOrRd")
    ax.set(title="Heatmap: hour vs weekday", xlabel="Day", ylabel="Hour")
    ax.set_xticks(range(7)); ax.set_xticklabels(DAYS)
    fig.colorbar(im, ax=ax, label="kWh")
    _save(fig, fig_dir / "04_heatmap.png")

    # Temperature relationship
    if "temperature_c" in df.columns:
        daily = df.resample("D").agg({c: "sum", "temperature_c": "mean"}).dropna()
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.scatter(daily["temperature_c"], daily[c], s=12, alpha=.6)
        ax.set(title="Daily consumption vs temperature",
               xlabel="Mean temperature (°C)", ylabel="Daily kWh")
        ax.grid(alpha=.3)
        _save(fig, fig_dir / "05_temperature.png")
        stats["temp_corr"] = float(daily.corr().iloc[0, 1])
    return stats


# ------------------------------------------------------------ 5. CLUSTERING
def cluster_days(df: pd.DataFrame, fig_dir: Path, k_range=range(2, 6)) -> dict:
    """Group days with similar 24-hour load shapes using KMeans."""
    daily = df.pivot_table(values="consumption_kwh", index=df.index.date,
                           columns="hour").dropna()
    best_k, best_s = None, -1
    for k in k_range:
        labels = KMeans(n_clusters=k, n_init=10, random_state=42).fit_predict(daily)
        s = silhouette_score(daily, labels)
        if s > best_s:
            best_k, best_s = k, s
    km = KMeans(n_clusters=best_k, n_init=10, random_state=42).fit(daily)

    fig, ax = plt.subplots(figsize=(8, 4))
    sizes = np.bincount(km.labels_)
    for i, centre in enumerate(km.cluster_centers_):
        ax.plot(range(24), centre, label=f"Cluster {i} ({sizes[i]} days)")
    ax.set(title=f"Daily load-profile clusters (k={best_k})",
           xlabel="Hour", ylabel="kWh", xticks=range(0, 24, 2))
    ax.legend(); ax.grid(alpha=.3)
    _save(fig, fig_dir / "06_clusters.png")
    return {"k": best_k, "silhouette": best_s, "sizes": sizes.tolist()}


# ------------------------------------------------------ 6. ANOMALY DETECTION
def detect_anomalies(df: pd.DataFrame, fig_dir: Path, threshold: float = 4.0):
    """Robust z-score (median/MAD) computed per (hour, weekend) group."""
    c = "consumption_kwh"
    g = df.groupby(["hour", "is_weekend"])[c]
    med = g.transform("median")
    mad = g.transform(lambda s: np.median(np.abs(s - s.median()))).replace(0, 1e-6)
    z = 0.6745 * (df[c] - med) / mad
    out = df[z > threshold][[c]].copy()
    out["robust_z"] = z[z > threshold].round(2)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(df.index, df[c], lw=.4, label="Consumption")
    ax.scatter(out.index, out[c], color="red", s=14, label="Anomaly", zorder=3)
    ax.set(title=f"Anomalies detected: {len(out)}", ylabel="kWh")
    ax.legend()
    _save(fig, fig_dir / "07_anomalies.png")
    return out


# -------------------------------------------------------------- 7. FORECAST
def _build_xy(df: pd.DataFrame):
    y = df["consumption_kwh"]
    X = pd.DataFrame(index=df.index)
    X["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    X["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    X["dayofweek"] = df["dayofweek"]
    X["month"] = df["month"]
    X["is_weekend"] = df["is_weekend"]
    X["lag_1"] = y.shift(1)
    X["lag_24"] = y.shift(24)
    X["lag_168"] = y.shift(168)
    X["roll_24"] = y.shift(1).rolling(24).mean()
    if "temperature_c" in df.columns:
        X["temperature_c"] = df["temperature_c"]
    X = X.dropna()
    return X, y.loc[X.index]


def forecast(df: pd.DataFrame, fig_dir: Path) -> dict:
    X, y = _build_xy(df)
    split = int(len(X) * 0.8)                       # chronological split
    Xtr, Xte, ytr, yte = X.iloc[:split], X.iloc[split:], y.iloc[:split], y.iloc[split:]

    models = {
        "LinearRegression": LinearRegression(),
        "RandomForest": RandomForestRegressor(n_estimators=100, n_jobs=-1,
                                              random_state=42),
    }
    results, preds = {}, {}
    for name, m in models.items():
        m.fit(Xtr, ytr)
        p = m.predict(Xte)
        preds[name] = p
        results[name] = {"MAE": mean_absolute_error(yte, p),
                         "RMSE": float(np.sqrt(mean_squared_error(yte, p))),
                         "R2": r2_score(yte, p)}

    best = min(results, key=lambda n: results[n]["RMSE"])
    last = 24 * 7
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(yte.index[-last:], yte.values[-last:], label="Actual")
    ax.plot(yte.index[-last:], preds[best][-last:], label=f"Predicted ({best})")
    ax.set(title="Forecast vs actual (last 7 days of test set)", ylabel="kWh")
    ax.legend(); ax.grid(alpha=.3)
    _save(fig, fig_dir / "08_forecast.png")
    return {"results": results, "best": best, "train": len(Xtr), "test": len(Xte)}
