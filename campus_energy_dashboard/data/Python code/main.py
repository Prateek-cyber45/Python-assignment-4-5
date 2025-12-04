from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


base = Path(__file__).parent
data_dir = base / "data"
out_dir = base / "outputs"
plots_dir = out_dir / "plots"

out_dir.mkdir(exist_ok=True)
plots_dir.mkdir(parents=True, exist_ok=True)

cleaned_csv = out_dir / "cleaned_energy_data.csv"
summary_csv = out_dir / "building_summary.csv"
summary_txt = out_dir / "summary.txt"
dashboard_png = plots_dir / "dashboard.png"


class MeterReading:
    def __init__(self, timestamp, kwh):
        self.timestamp = timestamp
        self.kwh = kwh


class Building:
    def __init__(self, name):
        self.name = name
        self.meter_readings = []

    def add_reading(self, reading):
        self.meter_readings.append(reading)

    def calculate_total_consumption(self):
        return float(sum(r.kwh for r in self.meter_readings))

    def generate_report(self):
        if not self.meter_readings:
            return {
                "building": self.name,
                "total_kwh": 0.0,
                "mean_kwh": 0.0,
                "max_kwh": 0.0,
            }
        vals = np.array([r.kwh for r in self.meter_readings], dtype=float)
        return {
            "building": self.name,
            "total_kwh": float(vals.sum()),
            "mean_kwh": float(vals.mean()),
            "max_kwh": float(vals.max()),
        }


class BuildingManager:
    def __init__(self):
        self.buildings = {}

    def get_or_create(self, name):
        if name not in self.buildings:
            self.buildings[name] = Building(name)
        return self.buildings[name]

    def load_from_df(self, df):
        for _, row in df.iterrows():
            name = row["building"]
            ts = row["timestamp"]
            kwh = float(row["kwh"])
            b = self.get_or_create(name)
            b.add_reading(MeterReading(ts, kwh))

    def summary_table(self):
        rows = [b.generate_report() for b in self.buildings.values()]
        return pd.DataFrame(rows)


def load_and_merge(data_path):
    frames = []
    log = []

    for csv_file in sorted(data_path.glob("*.csv")):
        try:
            df = pd.read_csv(csv_file, on_bad_lines="skip")
        except FileNotFoundError:
            log.append(f"Missing file: {csv_file.name}")
            continue

        if "timestamp" not in df.columns or "kwh" not in df.columns:
            log.append(f"Invalid structure in: {csv_file.name}")
            continue

        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df = df.dropna(subset=["timestamp"])
        df["kwh"] = pd.to_numeric(df["kwh"], errors="coerce")
        df = df.dropna(subset=["kwh"])

        if "building" not in df.columns:
            df["building"] = csv_file.stem

        frames.append(df)

    if not frames:
        raise ValueError("No valid CSV files found in data folder")

    merged = pd.concat(frames, ignore_index=True)
    merged = merged.sort_values("timestamp").reset_index(drop=True)

    for msg in log:
        print(msg)

    return merged


def calculate_daily_totals(df):
    df2 = df.copy()
    df2["timestamp"] = pd.to_datetime(df2["timestamp"])
    df2 = df2.set_index("timestamp")
    daily = df2.groupby("building")["kwh"].resample("D").sum().reset_index()
    return daily


def calculate_weekly_aggregates(df):
    df2 = df.copy()
    df2["timestamp"] = pd.to_datetime(df2["timestamp"])
    df2 = df2.set_index("timestamp")
    weekly = df2.groupby("building")["kwh"].resample("W").sum().reset_index()
    return weekly


def building_wise_summary(df):
    summary = df.groupby("building")["kwh"].agg(["mean", "min", "max", "sum"]).reset_index()
    summary = summary.rename(columns={"mean": "mean_kwh", "min": "min_kwh", "max": "max_kwh", "sum": "total_kwh"})
    return summary


def create_dashboard(daily, weekly, df):
    fig, axes = plt.subplots(3, 1, figsize=(10, 12))

    daily_pivot = daily.pivot(index="timestamp", columns="building", values="kwh")
    daily_pivot.plot(ax=axes[0])
    axes[0].set_title("Daily Consumption Trend")
    axes[0].set_xlabel("Date")
    axes[0].set_ylabel("kWh")

    weekly_mean = weekly.groupby("building")["kwh"].mean().reset_index()
    axes[1].bar(weekly_mean["building"], weekly_mean["kwh"])
    axes[1].set_title("Average Weekly Usage per Building")
    axes[1].set_xlabel("Building")
    axes[1].set_ylabel("kWh")

    idx = df.groupby("building")["kwh"].idxmax()
    peak = df.loc[idx]
    axes[2].scatter(peak["timestamp"], peak["kwh"])
    for _, row in peak.iterrows():
        axes[2].annotate(row["building"], (row["timestamp"], row["kwh"]), xytext=(5, 5), textcoords="offset points")
    axes[2].set_title("Peak Load per Building")
    axes[2].set_xlabel("Time")
    axes[2].set_ylabel("Peak kWh")

    plt.tight_layout()
    fig.savefig(dashboard_png)
    plt.close(fig)


def write_summary(df, building_summary, daily, weekly):
    total_campus = float(df["kwh"].sum())
    top_building_row = building_summary.sort_values("total_kwh", ascending=False).iloc[0]
    top_building = top_building_row["building"]
    top_building_kwh = float(top_building_row["total_kwh"])

    peak_idx = df["kwh"].idxmax()
    peak_row = df.loc[peak_idx]
    peak_time = peak_row["timestamp"]
    peak_building = peak_row["building"]
    peak_value = float(peak_row["kwh"])

    daily_trend = daily.groupby("timestamp")["kwh"].sum().reset_index()
    weekly_trend = weekly.groupby("timestamp")["kwh"].sum().reset_index()

    daily_peak_day = daily_trend.loc[daily_trend["kwh"].idxmax(), "timestamp"]
    weekly_peak_week = weekly_trend.loc[weekly_trend["kwh"].idxmax(), "timestamp"]

    lines = []
    lines.append("Campus Energy Usage Summary\n")
    lines.append(f"Generated: {datetime.now()}\n")
    lines.append(f"Total campus consumption: {total_campus:.2f} kWh\n")
    lines.append(f"Highest-consuming building: {top_building} ({top_building_kwh:.2f} kWh)\n")
    lines.append(f"Peak load time: {peak_time} in {peak_building} ({peak_value:.2f} kWh)\n")
    lines.append(f"Day with highest total load: {daily_peak_day}\n")
    lines.append(f"Week with highest total load (week ending): {weekly_peak_week}\n")

    summary_txt.write_text("\n".join(lines))

    print("\n".join(lines))


def main():
    df = load_and_merge(data_dir)

    daily_totals = calculate_daily_totals(df)
    weekly_totals = calculate_weekly_aggregates(df)
    building_summary = building_wise_summary(df)

    manager = BuildingManager()
    manager.load_from_df(df)
    building_summary_oop = manager.summary_table()

    building_summary.to_csv(summary_csv, index=False)
    df.to_csv(cleaned_csv, index=False)

    create_dashboard(daily_totals, weekly_totals, df)
    write_summary(df, building_summary_oop, daily_totals, weekly_totals)

    print("All outputs saved in 'outputs' folder.")


if __name__ == "__main__":
    main()
