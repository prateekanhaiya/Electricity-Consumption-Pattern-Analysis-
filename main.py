"""Run the complete Electricity Consumption Pattern Analysis pipeline.

Usage:
    python main.py                       # auto-generates sample data if missing
    python main.py --data my_data.csv    # use your own hourly CSV
"""
import argparse
from pathlib import Path

from src import analysis as an
from src.generate_data import generate


def main():
    ap = argparse.ArgumentParser(description="Electricity consumption analysis")
    ap.add_argument("--data", default="data/electricity_consumption.csv")
    ap.add_argument("--out", default="outputs")
    args = ap.parse_args()

    data_path, out = Path(args.data), Path(args.out)
    fig_dir = out / "figures"
    out.mkdir(exist_ok=True)

    if not data_path.exists():
        print(f"[1/7] {data_path} not found -> generating sample data")
        generate(out=str(data_path))
    else:
        print(f"[1/7] Using dataset: {data_path}")

    raw = an.load_data(str(data_path))
    df, n_missing = an.clean(raw)
    df = an.add_features(df)
    print(f"[2/7] Cleaned: {len(df)} rows ({n_missing} missing values handled)")

    stats = an.make_plots(df, fig_dir)
    print("[3/7] EDA plots saved")

    clusters = an.cluster_days(df, fig_dir)
    print(f"[4/7] Clustering done (k={clusters['k']})")

    anomalies = an.detect_anomalies(df, fig_dir)
    anomalies.to_csv(out / "anomalies.csv")
    print(f"[5/7] Anomalies found: {len(anomalies)}")

    fc = an.forecast(df, fig_dir)
    print("[6/7] Forecasting done")

    lines = [
        "ELECTRICITY CONSUMPTION PATTERN ANALYSIS - SUMMARY",
        "=" * 50,
        f"Period            : {df.index.min()} -> {df.index.max()}",
        f"Records           : {len(df)}",
        f"Missing handled   : {n_missing}",
        f"Total consumption : {df['consumption_kwh'].sum():.1f} kWh",
        f"Average per hour  : {df['consumption_kwh'].mean():.3f} kWh",
        f"Peak hour of day  : {stats['peak_hour']}:00",
        f"Lowest hour       : {stats['lowest_hour']}:00",
        f"Peak month        : {stats['peak_month']}",
    ]
    if "temp_corr" in stats:
        lines.append(f"Temp-usage corr   : {stats['temp_corr']:.2f}")
    lines += [
        f"Day-profile clusters: k={clusters['k']} "
        f"(silhouette={clusters['silhouette']:.2f}, sizes={clusters['sizes']})",
        f"Anomalies detected: {len(anomalies)}",
        "",
        f"Forecast (train={fc['train']}, test={fc['test']})",
    ]
    for name, m in fc["results"].items():
        lines.append(f"  {name:17s} MAE={m['MAE']:.3f}  RMSE={m['RMSE']:.3f}  R2={m['R2']:.3f}")
    lines.append(f"  Best model: {fc['best']}")

    report = "\n".join(lines)
    (out / "summary.txt").write_text(report, encoding="utf-8")
    print("[7/7] Summary saved\n")
    print(report)


if __name__ == "__main__":
    main()
