"""Bank-full schema and chronological partitions."""
import numpy as np
import pandas as pd
from .common import ROOT, download_member

URL = "https://archive.ics.uci.edu/static/public/222/bank+marketing.zip"
NUMERIC = ["age", "balance", "day", "campaign", "pdays", "previous"]
CATEGORICAL = ["job", "marital", "education", "default", "housing", "loan",
               "contact", "month", "poutcome"]
FEATURES = NUMERIC + CATEGORICAL

def obtain_data():
    return download_member(URL, ROOT / "data/raw/bank-full.csv",
                           "bank-full.csv", inner_archive="bank.zip")

def validate_features(frame):
    missing = sorted(set(FEATURES) - set(frame.columns))
    if missing:
        raise ValueError(f"Missing input columns: {missing}")
    result = frame[FEATURES].copy()
    for name in NUMERIC:
        result[name] = pd.to_numeric(result[name], errors="raise")
        if np.isinf(result[name].to_numpy(dtype=float)).any():
            raise ValueError(f"Infinite values in {name}.")
    for name in CATEGORICAL:
        result[name] = result[name].fillna("unknown").astype(str)
    return result

def load_data(path, mode="full"):
    frame = pd.read_csv(path, sep=";")
    if "y" not in frame or not frame["y"].isin(["yes", "no"]).all():
        raise ValueError("The y column must contain only yes and no.")
    validate_features(frame)
    frame = frame.assign(source_row=np.arange(len(frame)))
    frame = frame.drop_duplicates(subset=FEATURES + ["y"], keep="first")
    if mode == "quick" and len(frame) > 6000:
        frame = frame.iloc[np.linspace(0, len(frame) - 1, 6000, dtype=int)]
    frame = frame.reset_index(drop=True)
    if len(frame) < 200:
        raise ValueError("At least 200 rows are required for five chronological splits.")
    return frame

def partition(frame):
    n = len(frame)
    cuts = [0, int(n * .60), int(n * .75), int(n * .80), int(n * .85), n]
    names = ["train", "validation", "calibration", "threshold", "test"]
    splits = {name: frame.iloc[a:b].copy()
              for name, a, b in zip(names, cuts[:-1], cuts[1:])}
    for name, part in splits.items():
        if part["y"].nunique() != 2:
            raise ValueError(f"{name} needs both target classes. Use a larger dataset.")
    return splits
