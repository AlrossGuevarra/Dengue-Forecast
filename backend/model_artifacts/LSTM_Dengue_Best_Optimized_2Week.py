# ============================================================
# OPTIMIZED BEST LSTM VERSION FOR DENGUE RISK CLASSIFICATION
# 2-WEEK AHEAD VERSION
# Thesis: Web-Based System for Early Forecasting of Dengue Outbreak
#
# BASE MODEL
# - This starts from the user's BEST LSTM family: the earlier direct LSTM
#   that performed better than the hybrid and later unstable variants.
#
# KEY OPTIMIZATIONS
# 1) Direct multiclass classification: NO_RISK / LOW_RISK / HIGH_RISK
# 2) Keep SEQ_LEN = 12 (best among prior runs)
# 3) Reduce feature space from 106 to a cleaner LSTM-focused subset
# 4) Use moderate class weights: [1.0, 1.2, 3.0]
# 5) Use plain CrossEntropyLoss (more stable than aggressive focal setups)
# 6) Use simple LSTM + last hidden state (more stable than attention here)
# 7) Tune thresholds with a balanced score:
#       50% HIGH_RISK F1
#       25% HIGH_RISK recall
#       15% HIGH_RISK precision
#       10% balanced accuracy
# 8) Longer patience for more stable model selection
#
# EXPECTED EFFECT
# - Better balance between HIGH_RISK recall and precision
# - Better HIGH_RISK F1 than the over-conservative stabilized version
# - More stable than the aggressive later LSTM variants
#
# OUTPUT FOLDER
# - lstm_best_optimized_outputs/
# ============================================================

import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
import json
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from sklearn.preprocessing import StandardScaler


# -----------------------------
# USER SETTINGS
# -----------------------------
DATA_PATH = r"Final Dataset.csv"
OUTPUT_DIR = Path("lstm_best_optimized_2week_outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

RANDOM_STATE = 42

TRAIN_YEARS = [2021, 2022, 2023]
VAL_YEARS = [2024]
TEST_YEARS = [2025]

NO_RISK_LABEL = "NO_RISK"
LOW_RISK_LABEL = "LOW_RISK"
HIGH_RISK_LABEL = "HIGH_RISK"

RISK_LABELS = [NO_RISK_LABEL, LOW_RISK_LABEL, HIGH_RISK_LABEL]
RISK_TO_INT = {NO_RISK_LABEL: 0, LOW_RISK_LABEL: 1, HIGH_RISK_LABEL: 2}
INT_TO_RISK = {v: k for k, v in RISK_TO_INT.items()}

SEQ_LEN = 12
BATCH_SIZE = 384
MAX_EPOCHS = 40
LEARNING_RATE = 0.001
HIDDEN_SIZE = 64
NUM_LAYERS = 2
DROPOUT = 0.25
PATIENCE = 10
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# -----------------------------
# REPRODUCIBILITY
# -----------------------------
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(RANDOM_STATE)


# -----------------------------
# HELPERS
# -----------------------------
def print_section(title: str):
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


def safe_upper_strip(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.upper()


def make_week_cyclical(df: pd.DataFrame, week_col: str = "MorbidityWeek") -> pd.DataFrame:
    df["week_sin"] = np.sin(2 * np.pi * df[week_col] / 52.0)
    df["week_cos"] = np.cos(2 * np.pi * df[week_col] / 52.0)
    return df


def add_zero_streak(df: pd.DataFrame, group_col: str, case_col: str) -> pd.Series:
    values = []
    for _, sub in df.groupby(group_col, sort=False):
        streak = 0
        out = []
        for x in sub[case_col].values:
            out.append(streak)
            if x == 0:
                streak += 1
            else:
                streak = 0
        values.extend(out)
    return pd.Series(values, index=df.index)


def add_expanding_sameweek_mean(df: pd.DataFrame, group_cols, value_col: str) -> pd.Series:
    return df.groupby(group_cols)[value_col].transform(lambda s: s.shift(1).expanding().mean())


def add_expanding_sameweek_std(df: pd.DataFrame, group_cols, value_col: str) -> pd.Series:
    return df.groupby(group_cols)[value_col].transform(lambda s: s.shift(1).expanding().std())


def risk_from_threshold(rate_value: float, base_mean: float, base_std: float) -> str:
    high_thr = base_mean + 1.5 * base_std
    if rate_value > high_thr:
        return HIGH_RISK_LABEL
    elif rate_value >= base_mean:
        return LOW_RISK_LABEL
    else:
        return NO_RISK_LABEL


def per_label_metrics(y_true_str, y_pred_str, label: str):
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true_str,
        y_pred_str,
        labels=[label],
        zero_division=0
    )
    return {
        f"{label.lower()}_precision": float(precision[0]),
        f"{label.lower()}_recall": float(recall[0]),
        f"{label.lower()}_f1": float(f1[0]),
        f"{label.lower()}_support": int(support[0]),
    }


def build_recent_surge_features(df: pd.DataFrame) -> pd.DataFrame:
    eps = 1e-6

    df["cases_diff_1_2"] = df["cases_lag_1"] - df["cases_lag_2"]
    df["cases_diff_1_4mean"] = df["cases_lag_1"] - df["cases_rollmean_4"]
    df["cases_diff_1_8mean"] = df["cases_lag_1"] - df["cases_rollmean_8"]

    df["rate_diff_1_2"] = df["rate_lag_1"] - df["rate_lag_2"]
    df["rate_diff_1_4mean"] = df["rate_lag_1"] - df["rate_rollmean_4"]
    df["rate_diff_1_8mean"] = df["rate_lag_1"] - df["rate_rollmean_8"]

    df["cases_growth_1_over_2"] = df["cases_lag_1"] / (df["cases_lag_2"] + 1.0)
    df["cases_growth_1_over_4mean"] = df["cases_lag_1"] / (df["cases_rollmean_4"] + 1.0)

    df["rate_growth_1_over_2"] = df["rate_lag_1"] / (df["rate_lag_2"] + eps)
    df["rate_growth_1_over_4mean"] = df["rate_lag_1"] / (df["rate_rollmean_4"] + eps)

    df["rain_recent_change"] = df["rain_lag_1"] - df["rain_rollmean_4"]
    df["temp_recent_change"] = df["temp_lag_1"] - df["temp_rollmean_4"]
    df["hum_recent_change"] = df["hum_lag_1"] - df["hum_rollmean_4"]

    df["recent_spike_cases_2x4"] = (df["cases_lag_1"] > (2.0 * (df["cases_rollmean_4"] + 1.0))).astype(int)
    df["recent_spike_rate_2x4"] = (df["rate_lag_1"] > (2.0 * (df["rate_rollmean_4"] + eps))).astype(int)

    return df


class SequenceDataset(Dataset):
    def __init__(self, X_seq, y):
        self.X_seq = torch.tensor(X_seq, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X_seq[idx], self.y[idx]


class LSTMClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, dropout):
        super().__init__()
        lstm_dropout = dropout if num_layers > 1 else 0.0

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=lstm_dropout,
            bidirectional=False,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc_out = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out, _ = self.lstm(x)
        last_hidden = out[:, -1, :]
        z = self.dropout(last_hidden)
        z = self.fc1(z)
        z = self.relu(z)
        z = self.dropout(z)
        logits = self.fc_out(z)
        return logits


def evaluate_model(model, dataloader, device):
    model.eval()
    all_probs = []
    all_preds = []
    all_true = []

    with torch.no_grad():
        for xb, yb in dataloader:
            xb = xb.to(device)
            yb = yb.to(device)

            logits = model(xb)
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)

            all_probs.append(probs.cpu().numpy())
            all_preds.append(preds.cpu().numpy())
            all_true.append(yb.cpu().numpy())

    return np.concatenate(all_true), np.concatenate(all_preds), np.vstack(all_probs)


def tune_probability_rules(val_probs, y_val_true_int):
    """
    Balanced threshold tuning for the best LSTM family.
    """
    best = None
    best_score = -1.0

    y_val_true_str = pd.Series(y_val_true_int).map(INT_TO_RISK).values
    argmax_pred = np.argmax(val_probs, axis=1)

    high_thresholds = np.arange(0.40, 0.81, 0.05)
    low_thresholds = np.arange(0.20, 0.61, 0.05)

    for thr_high in high_thresholds:
        for thr_low in low_thresholds:
            pred = argmax_pred.copy()

            high_mask = val_probs[:, 2] >= thr_high
            low_mask = (val_probs[:, 1] >= thr_low) & (~high_mask)

            pred[high_mask] = 2
            pred[low_mask] = 1

            pred_str = pd.Series(pred).map(INT_TO_RISK).values

            bal_acc = balanced_accuracy_score(y_val_true_int, pred)
            hr = per_label_metrics(y_val_true_str, pred_str, HIGH_RISK_LABEL)

            hr_f1 = hr["high_risk_f1"]
            hr_recall = hr["high_risk_recall"]
            hr_precision = hr["high_risk_precision"]

            score = (
                0.50 * hr_f1 +
                0.25 * hr_recall +
                0.15 * hr_precision +
                0.10 * bal_acc
            )

            if score > best_score:
                best_score = score
                best = {
                    "thr_high": float(thr_high),
                    "thr_low": float(thr_low),
                    "score": float(score),
                    "balanced_accuracy": float(bal_acc),
                    "high_risk_f1": float(hr_f1),
                    "high_risk_precision": float(hr_precision),
                    "high_risk_recall": float(hr_recall),
                }

    return best


def apply_probability_rules(probs, thr_high, thr_low):
    pred = np.argmax(probs, axis=1)
    high_mask = probs[:, 2] >= thr_high
    low_mask = (probs[:, 1] >= thr_low) & (~high_mask)
    pred[high_mask] = 2
    pred[low_mask] = 1
    return pred


def build_sequences(df_part, feature_cols, seq_len):
    X_seq = []
    y = []
    meta_rows = []

    for _, sub in df_part.groupby("LocationKey", sort=False):
        sub = sub.sort_values(["Year", "MorbidityWeek"]).reset_index(drop=True)

        if len(sub) < seq_len:
            continue

        feature_matrix = sub[feature_cols].values.astype(np.float32)
        target_vec = sub["target_risk_int"].values.astype(int)

        for idx in range(seq_len - 1, len(sub)):
            X_seq.append(feature_matrix[idx - seq_len + 1: idx + 1])
            y.append(target_vec[idx])
            meta_rows.append(sub.iloc[idx])

    return np.array(X_seq, dtype=np.float32), np.array(y, dtype=np.int64), pd.DataFrame(meta_rows).reset_index(drop=True)


# -----------------------------
# LOAD DATA
# -----------------------------
print_section("1) LOAD DATA")
df = pd.read_csv(DATA_PATH)

required_cols = [
    "Year","MorbidityWeek","Barangay","Municipality","DengueCases",
    "RAINFALL_mm","TEMP_AVG_C","RELATIVE_HUMIDITY","Population"
]

missing_cols = [c for c in required_cols if c not in df.columns]
if missing_cols:
    raise ValueError(f"Missing required columns: {missing_cols}")

df["Barangay"] = safe_upper_strip(df["Barangay"])
df["Municipality"] = safe_upper_strip(df["Municipality"])

for col in ["Year","MorbidityWeek","DengueCases","RAINFALL_mm","TEMP_AVG_C","RELATIVE_HUMIDITY","Population"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

rows_before = len(df)
df = df.dropna(subset=required_cols).copy()

print(f"Rows loaded: {rows_before:,}")
print(f"Rows after dropping invalid required values: {len(df):,}")

if not df["MorbidityWeek"].between(1, 52).all():
    bad_weeks = sorted(df.loc[~df["MorbidityWeek"].between(1, 52), "MorbidityWeek"].unique().tolist())
    raise ValueError(f"Found invalid MorbidityWeek values: {bad_weeks}")

df["LocationKey"] = df["Municipality"] + " | " + df["Barangay"]

# -----------------------------
# FIX INVALID POPULATION
# -----------------------------
print_section("2) FIX INVALID POPULATION")
invalid_pop_mask = df["Population"] <= 0
print(f"Rows with Population <= 0 before fix: {int(invalid_pop_mask.sum()):,}")

location_positive_pop = (
    df.loc[df["Population"] > 0]
      .groupby("LocationKey")["Population"]
      .median()
)

fillable_mask = invalid_pop_mask & df["LocationKey"].isin(location_positive_pop.index)
df.loc[fillable_mask, "Population"] = df.loc[fillable_mask, "LocationKey"].map(location_positive_pop)

invalid_pop_mask = df["Population"] <= 0
remaining_invalid = int(invalid_pop_mask.sum())
print(f"Rows with Population <= 0 after location fix: {remaining_invalid:,}")

if remaining_invalid > 0:
    removed_df = df.loc[invalid_pop_mask].copy()
    removed_df.to_csv(OUTPUT_DIR / "removed_invalid_population_rows.csv", index=False)
    print("Rows removed because no valid population exists for that location.")
    df = df.loc[~invalid_pop_mask].copy()

print(f"Rows remaining after population fix/removal: {len(df):,}")

# -----------------------------
# DEDUPLICATE
# -----------------------------
print_section("3) DEDUPLICATE")
exact_dups = int(df.duplicated().sum())
print(f"Exact duplicate rows found: {exact_dups:,}")
if exact_dups > 0:
    df = df.drop_duplicates().copy()

key_cols = ["Year", "MorbidityWeek", "Municipality", "Barangay"]
dup_key_count = int(df.duplicated(subset=key_cols).sum())
print(f"Duplicate location-week keys before aggregation: {dup_key_count:,}")

df = (
    df.groupby(key_cols, as_index=False)
      .agg({
          "DengueCases": "sum",
          "RAINFALL_mm": "mean",
          "TEMP_AVG_C": "mean",
          "RELATIVE_HUMIDITY": "mean",
          "Population": "max",
      })
)

df["LocationKey"] = df["Municipality"] + " | " + df["Barangay"]
print(f"Duplicate location-week keys after aggregation: {int(df.duplicated(subset=key_cols).sum()):,}")

# -----------------------------
# BASIC SCAN
# -----------------------------
print_section("4) BASIC SCAN")
print(f"Rows: {len(df):,}")
print(f"Years: {sorted(df['Year'].unique().tolist())}")
print(f"Municipalities: {df['Municipality'].nunique():,}")
print(f"Unique barangay names: {df['Barangay'].nunique():,}")
print(f"Unique municipality+barangay locations: {df['LocationKey'].nunique():,}")
print(f"Zero-case rows: {(df['DengueCases'] == 0).sum():,}")
print(f"Non-zero rows: {(df['DengueCases'] > 0).sum():,}")

# -----------------------------
# FEATURE ENGINEERING
# -----------------------------
print_section("5) FEATURE ENGINEERING")
df = df.sort_values(["LocationKey", "Year", "MorbidityWeek"]).reset_index(drop=True)
df["CasesPer1000"] = (df["DengueCases"] / df["Population"]) * 1000.0
df = make_week_cyclical(df, "MorbidityWeek")
g_loc = df.groupby("LocationKey", group_keys=False)

case_lags = [1, 2, 3, 4, 8, 12]
weather_lags = [1, 2, 3, 4]

for lag in case_lags:
    df[f"cases_lag_{lag}"] = g_loc["DengueCases"].shift(lag)
    df[f"rate_lag_{lag}"] = g_loc["CasesPer1000"].shift(lag)

for lag in weather_lags:
    df[f"rain_lag_{lag}"] = g_loc["RAINFALL_mm"].shift(lag)
    df[f"temp_lag_{lag}"] = g_loc["TEMP_AVG_C"].shift(lag)
    df[f"hum_lag_{lag}"] = g_loc["RELATIVE_HUMIDITY"].shift(lag)

for window in [2, 4, 8]:
    df[f"cases_rollsum_{window}"] = g_loc["DengueCases"].transform(lambda s: s.shift(1).rolling(window=window, min_periods=1).sum())
    df[f"cases_rollmean_{window}"] = g_loc["DengueCases"].transform(lambda s: s.shift(1).rolling(window=window, min_periods=1).mean())
    df[f"rate_rollmean_{window}"] = g_loc["CasesPer1000"].transform(lambda s: s.shift(1).rolling(window=window, min_periods=1).mean())
    df[f"rate_rollmax_{window}"] = g_loc["CasesPer1000"].transform(lambda s: s.shift(1).rolling(window=window, min_periods=1).max())
    df[f"rain_rollmean_{window}"] = g_loc["RAINFALL_mm"].transform(lambda s: s.shift(1).rolling(window=window, min_periods=1).mean())
    df[f"temp_rollmean_{window}"] = g_loc["TEMP_AVG_C"].transform(lambda s: s.shift(1).rolling(window=window, min_periods=1).mean())
    df[f"hum_rollmean_{window}"] = g_loc["RELATIVE_HUMIDITY"].transform(lambda s: s.shift(1).rolling(window=window, min_periods=1).mean())

df["loc_hist_mean_rate"] = g_loc["CasesPer1000"].transform(lambda s: s.shift(1).expanding().mean())
df["loc_hist_std_rate"] = g_loc["CasesPer1000"].transform(lambda s: s.shift(1).expanding().std())
df["loc_hist_max_rate"] = g_loc["CasesPer1000"].transform(lambda s: s.shift(1).expanding().max())
df["loc_hist_mean_cases"] = g_loc["DengueCases"].transform(lambda s: s.shift(1).expanding().mean())

df["loc_sameweek_mean_rate"] = add_expanding_sameweek_mean(df, ["LocationKey", "MorbidityWeek"], "CasesPer1000")
df["mun_sameweek_mean_rate"] = add_expanding_sameweek_mean(df, ["Municipality", "MorbidityWeek"], "CasesPer1000")

mun_week = (
    df.groupby(["Municipality", "Year", "MorbidityWeek"], as_index=False)
      .agg({
          "DengueCases": "sum",
          "CasesPer1000": "mean",
          "RAINFALL_mm": "mean",
      })
      .sort_values(["Municipality", "Year", "MorbidityWeek"])
      .reset_index(drop=True)
)

g_mun = mun_week.groupby("Municipality", group_keys=False)
mun_week["mun_cases_lag_1"] = g_mun["DengueCases"].shift(1)
mun_week["mun_rate_lag_1"] = g_mun["CasesPer1000"].shift(1)
mun_week["mun_rate_rollmean_4"] = g_mun["CasesPer1000"].transform(lambda s: s.shift(1).rolling(window=4, min_periods=1).mean())

df = df.merge(
    mun_week[[
        "Municipality", "Year", "MorbidityWeek",
        "mun_cases_lag_1", "mun_rate_lag_1", "mun_rate_rollmean_4"
    ]],
    on=["Municipality", "Year", "MorbidityWeek"],
    how="left"
)

df["zero_streak"] = add_zero_streak(df, "LocationKey", "DengueCases")
df["rain_x_humidity"] = df["RAINFALL_mm"] * df["RELATIVE_HUMIDITY"]
df["temp_x_humidity"] = df["TEMP_AVG_C"] * df["RELATIVE_HUMIDITY"]
df["cases_over_loc_mean"] = df["DengueCases"] / (df["loc_hist_mean_cases"] + 1.0)
df["rate_over_loc_mean"] = df["CasesPer1000"] / (df["loc_hist_mean_rate"] + 1e-6)

df = build_recent_surge_features(df)

# -----------------------------
# BUILD TARGET
# -----------------------------
print_section("6) BUILD DIRECT RISK LABEL TARGET (2-WEEK AHEAD)")
df["target_rate_next_2week"] = g_loc["CasesPer1000"].shift(-2)
df["baseline_mean_for_next"] = df["loc_hist_mean_rate"]
df["baseline_std_for_next"] = df["loc_hist_std_rate"].fillna(0.0)
df["baseline_high_thr_for_next"] = df["baseline_mean_for_next"] + (1.5 * df["baseline_std_for_next"])

df["target_risk_next_2week"] = df.apply(
    lambda row: risk_from_threshold(
        rate_value=row["target_rate_next_2week"],
        base_mean=row["baseline_mean_for_next"],
        base_std=row["baseline_std_for_next"],
    ) if pd.notna(row["target_rate_next_2week"]) else np.nan,
    axis=1
)

for c in df.columns:
    if any(k in c for k in ["lag_", "roll", "hist_", "sameweek", "_diff_", "_growth_", "spike", "mean", "std"]):
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)

# -----------------------------
# BUILD MODEL DATASET
# -----------------------------
print_section("7) BUILD MODEL DATASET")

# Reduced LSTM-focused feature set
feature_cols = [
    "MorbidityWeek","week_sin","week_cos",
    "Population",
    "RAINFALL_mm","TEMP_AVG_C","RELATIVE_HUMIDITY",
    "rain_x_humidity","temp_x_humidity",
    "zero_streak",
    "loc_hist_mean_rate","loc_hist_std_rate","loc_hist_max_rate",
    "loc_hist_mean_cases",
    "loc_sameweek_mean_rate","mun_sameweek_mean_rate",
    "mun_cases_lag_1","mun_rate_lag_1","mun_rate_rollmean_4",
    "cases_over_loc_mean","rate_over_loc_mean",
    "cases_diff_1_2","cases_diff_1_4mean",
    "rate_diff_1_2","rate_diff_1_4mean",
    "cases_growth_1_over_2","cases_growth_1_over_4mean",
    "rate_growth_1_over_2","rate_growth_1_over_4mean",
    "rain_recent_change","temp_recent_change","hum_recent_change",
    "recent_spike_cases_2x4","recent_spike_rate_2x4",
]

feature_cols += [f"cases_lag_{lag}" for lag in case_lags]
feature_cols += [f"rate_lag_{lag}" for lag in case_lags]
feature_cols += [f"rain_lag_{lag}" for lag in weather_lags]
feature_cols += [f"temp_lag_{lag}" for lag in weather_lags]
feature_cols += [f"hum_lag_{lag}" for lag in weather_lags]
feature_cols += [f"cases_rollsum_{w}" for w in [2,4,8]]
feature_cols += [f"cases_rollmean_{w}" for w in [2,4,8]]
feature_cols += [f"rate_rollmean_{w}" for w in [2,4,8]]
feature_cols += [f"rate_rollmax_{w}" for w in [2,4,8]]
feature_cols += [f"rain_rollmean_{w}" for w in [2,4,8]]
feature_cols += [f"temp_rollmean_{w}" for w in [2,4,8]]
feature_cols += [f"hum_rollmean_{w}" for w in [2,4,8]]

feature_cols = list(dict.fromkeys(feature_cols))

model_df = df.dropna(subset=["target_risk_next_2week"]).copy()
for c in feature_cols:
    model_df[c] = pd.to_numeric(model_df[c], errors="coerce").fillna(0.0)

model_df["target_risk_int"] = model_df["target_risk_next_2week"].map(RISK_TO_INT).astype(int)

print(f"Rows available for modeling before sequence building: {len(model_df):,}")
print(f"Number of features: {len(feature_cols)}")

train_df = model_df.loc[model_df["Year"].isin(TRAIN_YEARS)].copy()
val_df = model_df.loc[model_df["Year"].isin(VAL_YEARS)].copy()
test_df = model_df.loc[model_df["Year"].isin(TEST_YEARS)].copy()

print(f"Train rows: {len(train_df):,}")
print(f"Validation rows: {len(val_df):,}")
print(f"Test rows: {len(test_df):,}")

# -----------------------------
# SCALE FEATURES
# -----------------------------
print_section("8) SCALE FEATURES")
scaler = StandardScaler()
scaler.fit(train_df[feature_cols].values)

train_scaled = pd.DataFrame(scaler.transform(train_df[feature_cols].values), columns=feature_cols, index=train_df.index, dtype=np.float32)
val_scaled = pd.DataFrame(scaler.transform(val_df[feature_cols].values), columns=feature_cols, index=val_df.index, dtype=np.float32)
test_scaled = pd.DataFrame(scaler.transform(test_df[feature_cols].values), columns=feature_cols, index=test_df.index, dtype=np.float32)

for c in feature_cols:
    train_df[c] = train_scaled[c].astype(np.float32)
    val_df[c] = val_scaled[c].astype(np.float32)
    test_df[c] = test_scaled[c].astype(np.float32)

with open(OUTPUT_DIR / "lstm_scaler_stats.json", "w", encoding="utf-8") as f:
    json.dump({
        "feature_cols": feature_cols,
        "mean": scaler.mean_.tolist(),
        "scale": scaler.scale_.tolist(),
    }, f)

# -----------------------------
# BUILD SEQUENCES
# -----------------------------
print_section("9) BUILD SEQUENCES")
X_train_seq, y_train, train_meta = build_sequences(train_df, feature_cols, SEQ_LEN)
X_val_seq, y_val, val_meta = build_sequences(val_df, feature_cols, SEQ_LEN)
X_test_seq, y_test, test_meta = build_sequences(test_df, feature_cols, SEQ_LEN)

print(f"Train sequences: {len(X_train_seq):,}")
print(f"Validation sequences: {len(X_val_seq):,}")
print(f"Test sequences: {len(X_test_seq):,}")
print(f"Sequence shape: {X_train_seq.shape}")

# -----------------------------
# DATALOADERS
# -----------------------------
print_section("10) DATALOADER SETUP")
train_dataset = SequenceDataset(X_train_seq, y_train)
val_dataset = SequenceDataset(X_val_seq, y_val)
test_dataset = SequenceDataset(X_test_seq, y_test)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=False)

class_counts = np.bincount(y_train, minlength=3)
print("Training class counts after sequence building:")
for i, c in enumerate(class_counts):
    print(f"{INT_TO_RISK[i]}: {int(c)}")

# -----------------------------
# LOSS SETUP
# -----------------------------
print_section("11) LOSS SETUP")
weights = np.array([1.0, 1.2, 3.0], dtype=np.float32)
print("Class weights used:")
for i, w in enumerate(weights):
    print(f"{INT_TO_RISK[i]}: {float(w):.6f}")

criterion = nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32).to(DEVICE))

# -----------------------------
# MODEL SETUP
# -----------------------------
print_section("12) INITIALIZE MODEL")
input_size = len(feature_cols)
model = LSTMClassifier(
    input_size=input_size,
    hidden_size=HIDDEN_SIZE,
    num_layers=NUM_LAYERS,
    num_classes=3,
    dropout=DROPOUT,
).to(DEVICE)

optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="max", factor=0.5, patience=2
)

print(model)
print(f"\nUsing device: {DEVICE}")

# -----------------------------
# TRAIN MODEL
# -----------------------------
print_section("13) TRAIN MODEL")
best_state = None
best_epoch = None
best_thresholds = None
best_selection_score = -1.0
epochs_without_improve = 0
history_rows = []

for epoch in range(1, MAX_EPOCHS + 1):
    model.train()
    train_loss_sum = 0.0
    train_count = 0

    for xb, yb in train_loader:
        xb = xb.to(DEVICE)
        yb = yb.to(DEVICE)

        optimizer.zero_grad()
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        train_loss_sum += float(loss.item()) * len(yb)
        train_count += len(yb)

    train_loss = train_loss_sum / max(train_count, 1)

    y_val_true, _, y_val_probs = evaluate_model(model, val_loader, DEVICE)
    tuned = tune_probability_rules(y_val_probs, y_val_true)
    y_val_pred = apply_probability_rules(y_val_probs, tuned["thr_high"], tuned["thr_low"])

    val_acc = accuracy_score(y_val_true, y_val_pred)
    val_bal_acc = balanced_accuracy_score(y_val_true, y_val_pred)
    val_macro_f1 = f1_score(y_val_true, y_val_pred, average="macro")

    y_val_true_str = pd.Series(y_val_true).map(INT_TO_RISK).values
    y_val_pred_str = pd.Series(y_val_pred).map(INT_TO_RISK).values
    hr = per_label_metrics(y_val_true_str, y_val_pred_str, HIGH_RISK_LABEL)

    selection_score = tuned["score"]
    scheduler.step(selection_score)

    history_rows.append({
        "epoch": epoch,
        "train_loss": train_loss,
        "val_accuracy": val_acc,
        "val_balanced_accuracy": val_bal_acc,
        "val_macro_f1": val_macro_f1,
        "val_high_risk_precision": hr["high_risk_precision"],
        "val_high_risk_recall": hr["high_risk_recall"],
        "val_high_risk_f1": hr["high_risk_f1"],
        "thr_high": tuned["thr_high"],
        "thr_low": tuned["thr_low"],
        "selection_score": selection_score,
        "learning_rate": optimizer.param_groups[0]["lr"],
    })

    print(
        f"Epoch {epoch:02d} | "
        f"train_loss={train_loss:.6f} | "
        f"val_acc={val_acc:.6f} | "
        f"val_bal_acc={val_bal_acc:.6f} | "
        f"val_high_precision={hr['high_risk_precision']:.6f} | "
        f"val_high_recall={hr['high_risk_recall']:.6f} | "
        f"val_high_f1={hr['high_risk_f1']:.6f} | "
        f"thr_high={tuned['thr_high']:.2f} | thr_low={tuned['thr_low']:.2f}"
    )

    if selection_score > best_selection_score:
        best_selection_score = selection_score
        best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        best_epoch = epoch
        best_thresholds = tuned
        epochs_without_improve = 0
    else:
        epochs_without_improve += 1

    if epochs_without_improve >= PATIENCE:
        print(f"\nEarly stopping triggered at epoch {epoch}.")
        break

history_df = pd.DataFrame(history_rows)
history_df.to_csv(OUTPUT_DIR / "lstm_best_optimized_2week_training_history.csv", index=False)

print(f"\nBest epoch: {best_epoch}")
print("Best validation thresholds:")
print(best_thresholds)

model.load_state_dict(best_state)
torch.save(model.state_dict(), OUTPUT_DIR / "lstm_best_optimized_2week_model.pt")

# -----------------------------
# TEST EVALUATION
# -----------------------------
print_section("14) TEST EVALUATION")
y_test_true, _, y_test_probs = evaluate_model(model, test_loader, DEVICE)
y_test_pred = apply_probability_rules(
    y_test_probs,
    thr_high=best_thresholds["thr_high"],
    thr_low=best_thresholds["thr_low"]
)

y_test_true_str = pd.Series(y_test_true).map(INT_TO_RISK).values
y_test_pred_str = pd.Series(y_test_pred).map(INT_TO_RISK).values

acc = accuracy_score(y_test_true, y_test_pred)
bal_acc = balanced_accuracy_score(y_test_true, y_test_pred)
macro_f1 = f1_score(y_test_true, y_test_pred, average="macro")
cm = confusion_matrix(y_test_true, y_test_pred, labels=[0, 1, 2])

print(f"Overall Accuracy:         {acc:.6f}")
print(f"Balanced Accuracy:        {bal_acc:.6f}")
print(f"Macro F1:                {macro_f1:.6f}")
print("\nConfusion Matrix (rows=true, cols=pred):")
print(pd.DataFrame(cm, index=RISK_LABELS, columns=RISK_LABELS))

print("\nClassification Report:")
print(classification_report(
    y_test_true,
    y_test_pred,
    target_names=RISK_LABELS,
    digits=4,
    zero_division=0
))

hr = per_label_metrics(y_test_true_str, y_test_pred_str, HIGH_RISK_LABEL)
lr = per_label_metrics(y_test_true_str, y_test_pred_str, LOW_RISK_LABEL)
nr = per_label_metrics(y_test_true_str, y_test_pred_str, NO_RISK_LABEL)

print("\nHigh-risk specific metrics:")
for k, v in hr.items():
    print(f"{k}: {v}" if "support" in k else f"{k}: {v:.6f}")

print("\nLow-risk specific metrics:")
for k, v in lr.items():
    print(f"{k}: {v}" if "support" in k else f"{k}: {v:.6f}")

# -----------------------------
# SAVE OUTPUTS
# -----------------------------
print_section("15) SAVE OUTPUT FILES")
pred_df = test_meta.copy().reset_index(drop=True)
pred_df["true_risk"] = y_test_true_str
pred_df["pred_risk"] = y_test_pred_str
pred_df["proba_no_risk"] = y_test_probs[:, 0]
pred_df["proba_low_risk"] = y_test_probs[:, 1]
pred_df["proba_high_risk"] = y_test_probs[:, 2]

save_cols = [
    "Year","MorbidityWeek","Municipality","Barangay","LocationKey","Population","DengueCases",
    "CasesPer1000","target_rate_next_2week","baseline_mean_for_next","baseline_std_for_next",
    "baseline_high_thr_for_next","true_risk","pred_risk","proba_no_risk","proba_low_risk","proba_high_risk"
]
pred_df[save_cols].to_csv(OUTPUT_DIR / "lstm_best_optimized_2week_test_predictions.csv", index=False)

metrics_rows = [
    {"metric": "accuracy", "value": acc},
    {"metric": "balanced_accuracy", "value": bal_acc},
    {"metric": "macro_f1", "value": macro_f1},
    {"metric": "high_risk_precision", "value": hr["high_risk_precision"]},
    {"metric": "high_risk_recall", "value": hr["high_risk_recall"]},
    {"metric": "high_risk_f1", "value": hr["high_risk_f1"]},
    {"metric": "low_risk_precision", "value": lr["low_risk_precision"]},
    {"metric": "low_risk_recall", "value": lr["low_risk_recall"]},
    {"metric": "low_risk_f1", "value": lr["low_risk_f1"]},
    {"metric": "no_risk_precision", "value": nr["no_risk_precision"]},
    {"metric": "no_risk_recall", "value": nr["no_risk_recall"]},
    {"metric": "no_risk_f1", "value": nr["no_risk_f1"]},
    {"metric": "best_epoch", "value": best_epoch},
    {"metric": "thr_high", "value": best_thresholds["thr_high"]},
    {"metric": "thr_low", "value": best_thresholds["thr_low"]},
    {"metric": "sequence_length", "value": SEQ_LEN},
    {"metric": "feature_count", "value": len(feature_cols)},
]
pd.DataFrame(metrics_rows).to_csv(OUTPUT_DIR / "lstm_best_optimized_2week_metrics.csv", index=False)
pd.DataFrame(cm, index=RISK_LABELS, columns=RISK_LABELS).to_csv(OUTPUT_DIR / "lstm_best_optimized_2week_confusion_matrix.csv")
pd.DataFrame([best_thresholds]).to_csv(OUTPUT_DIR / "lstm_best_optimized_2week_best_thresholds.csv", index=False)

print(f"Saved: {OUTPUT_DIR / 'lstm_best_optimized_2week_training_history.csv'}")
print(f"Saved: {OUTPUT_DIR / 'lstm_best_optimized_2week_model.pt'}")
print(f"Saved: {OUTPUT_DIR / 'lstm_scaler_stats.json'}")
print(f"Saved: {OUTPUT_DIR / 'lstm_best_optimized_2week_test_predictions.csv'}")
print(f"Saved: {OUTPUT_DIR / 'lstm_best_optimized_2week_metrics.csv'}")
print(f"Saved: {OUTPUT_DIR / 'lstm_best_optimized_2week_confusion_matrix.csv'}")
print(f"Saved: {OUTPUT_DIR / 'lstm_best_optimized_2week_best_thresholds.csv'}")
if (OUTPUT_DIR / 'removed_invalid_population_rows.csv').exists():
    print(f"Saved: {OUTPUT_DIR / 'removed_invalid_population_rows.csv'}")

print_section("DONE")
print("Optimized best LSTM training and evaluation completed successfully.")
