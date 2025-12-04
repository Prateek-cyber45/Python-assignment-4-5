import pandas as pd
import numpy as np
from pathlib import Path

def generate_building_data(building_name, start_date, end_date,
                           base_kwh, peak_extra=5.0, noise=2.0):
    """
    Generate hourly kWh data for one building between start_date and end_date.
    - base_kwh: baseline consumption
    - peak_extra: extra kWh added during peak hours
    - noise: random variation
    """
    timestamps = pd.date_range(start=start_date, end=end_date, freq="H")

    kwh_values = []

    rng = np.random.default_rng(seed=42) 

    for ts in timestamps:
        hour = ts.hour

        if 8 <= hour <= 20:
            mean_kwh = base_kwh + peak_extra
        else:
            mean_kwh = base_kwh

        kwh = rng.normal(loc=mean_kwh, scale=noise)

        kwh = max(kwh, 0.1)

        kwh_values.append(round(float(kwh), 2))

    df = pd.DataFrame({
        "timestamp": timestamps,
        "kwh": kwh_values
    })

    return df


def main():
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)

    start_date = "2025-01-01 00:00"
    end_date = "2025-01-07 23:00"

    df_a = generate_building_data("Building_A", start_date, end_date,
                                  base_kwh=10.0, peak_extra=4.0, noise=1.5)
    df_b = generate_building_data("Building_B", start_date, end_date,
                                  base_kwh=20.0, peak_extra=6.0, noise=2.5)
    df_c = generate_building_data("Building_C", start_date, end_date,
                                  base_kwh=7.0, peak_extra=3.0, noise=1.0)

    df_a.to_csv(data_dir / "building_a.csv", index=False)
    df_b.to_csv(data_dir / "building_b.csv", index=False)
    df_c.to_csv(data_dir / "building_c.csv", index=False)

    print("Sample CSV files created in 'data/' folder:")
    print(" - building_a.csv")
    print(" - building_b.csv")
    print(" - building_c.csv")


if __name__ == "__main__":
    main()
