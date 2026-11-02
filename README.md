# ⚡ Electricity Consumption Pattern Analysis

A Python project that analyses hourly electricity-consumption data to discover **usage patterns**, **peak hours**, **abnormal spikes**, and to **forecast** future demand using machine learning.

## Features
- Data cleaning (duplicates, missing values, hourly re-indexing)
- Feature engineering (hour, weekday, month, weekend flag, lag features)
- Exploratory analysis with 5 plots (hourly profile, weekday, monthly trend, heatmap, temperature vs usage)
- **KMeans clustering** of daily load profiles (best `k` chosen by silhouette score)
- **Anomaly detection** using robust z-score (median/MAD) per hour-and-weekend group
- **Forecasting** next-hour consumption with Linear Regression vs Random Forest (MAE, RMSE, R²)
- Auto-generated text summary + saved figures

## Project Structure
```
Electricity-Consumption-Pattern-Analysis/
├── data/                  # dataset (sample is auto-generated)
├── src/
│   ├── generate_data.py   # synthetic dataset generator
│   ├── analysis.py        # cleaning, EDA, clustering, anomalies, forecasting
│   └── generate_campus_data.py  # campus dataset for the DA-018 notebook
├── outputs/               # figures/, summary.txt, anomalies.csv
├── notebooks/             # Energy_Analytics_DA018.ipynb (DA-018 brief)
├── main.py                # runs the whole pipeline
├── WORKFLOW.md            # step-by-step workflow
├── requirements.txt
└── README.md
```

## DA-018 Notebook: Campus Energy Analytics
`notebooks/Energy_Analytics_DA018.ipynb` (already executed, charts included) answers the facility-manager brief on a synthetic campus dataset (4 areas, 15-minute kWh readings, 2023):
- Confirms units and sampling interval, audits duplicates, negatives and missing intervals
- Fills only gaps of 1 hour or less; longer gaps are left missing and those days are excluded
- Daily and monthly summaries, weekday vs weekend comparison, area comparison
- Peak-demand periods and unusual-reading inspection (robust z-score events)
- 8 charts, a list of energy-saving questions, and a savings what-if with stated assumptions only (no cost estimate)

Data generator: `src/generate_campus_data.py` writes `data/campus_energy_readings.csv`.

## Getting Started
```bash
git clone https://github.com/<your-username>/Electricity-Consumption-Pattern-Analysis.git
cd Electricity-Consumption-Pattern-Analysis

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt

python main.py
jupyter notebook notebooks/Energy_Analytics_DA018.ipynb   # DA-018 notebook
```

### Use your own dataset
Provide a CSV with **hourly** data and these columns (`temperature_c` is optional):

| timestamp           | consumption_kwh | temperature_c |
|---------------------|-----------------|---------------|
| 2023-01-01 00:00:00 | 0.52            | 18.4          |

```bash
python main.py --data path/to/your_data.csv
```

## Sample Results
(on the included synthetic 1-year dataset)

| Metric | Value |
|---|---|
| Peak hour | 20:00 |
| Peak month | July |
| Temperature–usage correlation | 0.87 |
| Anomalies detected | 47 |
| Linear Regression | MAE 0.222 · RMSE 0.341 · R² 0.658 |
| **Random Forest** | **MAE 0.097 · RMSE 0.239 · R² 0.832** |

Plots are saved in `outputs/figures/`.

## Tech Stack
Python · Pandas · NumPy · Matplotlib · Scikit-learn

## Future Improvements
- Add LSTM / Prophet models
- Interactive dashboard (Streamlit)
- Support for multiple households / smart meters

## License
MIT
