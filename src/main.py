import pandas as pd
import numpy as np

df = pd.read_csv("./data/jena_climate_2009_2016.csv")

print(df.columns)

# Lean feature set — independent variables only
features = [
    "p (mbar)",      # pressure
    "T (degC)",      # temperature (target)
    "Tdew (degC)",   # dew point
    "rh (%)",        # relative humidity
    "wv (m/s)",      # wind speed
    "wd (deg)",      # wind direction
    "rho (g/m**3)"   # air density
]

df = df[features]
df["wd_sin"] = np.sin(np.deg2rad(df["wd (deg)"]))
df["wd_cos"] = np.cos(np.deg2rad(df["wd (deg)"]))
df = df.drop(columns=["wd (deg)"])
