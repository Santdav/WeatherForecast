import pandas as pd
import numpy as np

df = pd.read_csv("./data/jena_climate_2009_2016.csv")


#  Lean feature set — independent variables only
features = [
    "Date Time",
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

df["Date Time"]  = pd.to_datetime(df['Date Time'], format="%d.%m.%Y %H:%M:%S")
df.to_csv("./data/clean.csv", index=False)