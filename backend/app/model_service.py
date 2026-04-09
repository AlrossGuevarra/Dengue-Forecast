
from __future__ import annotations
import json, re, difflib
from functools import lru_cache
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

NO_RISK_LABEL = "NO_RISK"
LOW_RISK_LABEL = "LOW_RISK"
HIGH_RISK_LABEL = "HIGH_RISK"
INT_TO_RISK = {0: NO_RISK_LABEL, 1: LOW_RISK_LABEL, 2: HIGH_RISK_LABEL}
SEQ_LEN = 12
HIDDEN_SIZE = 64
NUM_LAYERS = 2
DROPOUT = 0.25
DEVICE = "cpu"
FORECAST_HORIZON_WEEKS = 2
ROMAN = {1:'I',2:'II',3:'III',4:'IV',5:'V',6:'VI',7:'VII',8:'VIII',9:'IX',10:'X',11:'XI',12:'XII'}
ISLA_VERDE_SPECIAL = {
    "LIPONPON": "LIPONPONISLAVERDE",
    "SANAGAPITO": "SANAGAPITOISLAVERDE",
    "SANAGUSTINKANLURAN": "SANAGUSTINKANLURANISLAVERDE",
    "SANAGUSTINSILANGAN": "SANAGUSTINSILANGANISLAVERDE",
    "SANANDRES": "SANANDRESISLAVERDE",
    "SANANTONIO": "SANANTONIOISLAVERDE",
}

def norm(s: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", str(s or "").upper().strip())

def norm_loose(s: str) -> str:
    s = str(s or "").upper()
    s = s.replace("BARANGAY", "").replace("BRGY", "").replace(",", "")
    s = s.replace("NORTE", "NORTH").replace("SUR", "SOUTH")
    s = re.sub(r"\s+", "", s)
    return re.sub(r"[^A-Z0-9]", "", s)

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
            streak = streak + 1 if x == 0 else 0
        values.extend(out)
    return pd.Series(values, index=df.index)

def add_expanding_sameweek_mean(df: pd.DataFrame, group_cols, value_col: str) -> pd.Series:
    return df.groupby(group_cols)[value_col].transform(lambda s: s.shift(1).expanding().mean())

def build_recent_surge_features(df: pd.DataFrame) -> pd.DataFrame:
    eps = 1e-6
    df["cases_diff_1_2"] = df["cases_lag_1"] - df["cases_lag_2"]
    df["cases_diff_1_4mean"] = df["cases_lag_1"] - df["cases_rollmean_4"]
    df["rate_diff_1_2"] = df["rate_lag_1"] - df["rate_lag_2"]
    df["rate_diff_1_4mean"] = df["rate_lag_1"] - df["rate_rollmean_4"]
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

class LSTMClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, dropout):
        super().__init__()
        lstm_dropout = dropout if num_layers > 1 else 0.0
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True, dropout=lstm_dropout, bidirectional=False)
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
        return self.fc_out(z)

class ForecastService:
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.data_path = project_root / "data" / "Final Dataset.csv"
        self.artifact_dir = project_root / "model_artifacts"
        scaler_stats = json.loads((self.artifact_dir / "lstm_scaler_stats.json").read_text(encoding="utf-8"))
        self.feature_cols = scaler_stats["feature_cols"]
        self.scaler_mean = np.array(scaler_stats["mean"], dtype=np.float32)
        self.scaler_scale = np.array(scaler_stats["scale"], dtype=np.float32)
        thresholds_df = pd.read_csv(self.artifact_dir / "lstm_best_optimized_2week_best_thresholds.csv")
        self.thr_high = float(thresholds_df.iloc[0]["thr_high"])
        self.thr_low = float(thresholds_df.iloc[0]["thr_low"])
        self.model = LSTMClassifier(len(self.feature_cols), HIDDEN_SIZE, NUM_LAYERS, 3, DROPOUT)
        self.model.load_state_dict(torch.load(self.artifact_dir / "lstm_best_optimized_2week_model.pt", map_location=DEVICE))
        self.model.eval()
        self._predict_cache = None
        self._summary_cache = None
        self._match_cache = None
        self.model_df = None
        self.raw_df = None
        self.boundary_pairs = set()
        self.boundary_by_muni = {}
        self.hist_lookup = {}
        self._load_dataset()
        self._load_boundary_pairs()
        self._build_hist_lookup()

    def _load_dataset(self):
        self.model_df, self.raw_df = self._prepare_dataset(pd.read_csv(self.data_path))

    def _load_boundary_pairs(self):
        boundary_path = self.project_root.parent / "frontend" / "public" / "batangas_barangays.geojson"
        with open(boundary_path, "r", encoding="utf-8") as f:
            geo = json.load(f)
        self.boundary_pairs = set()
        self.boundary_by_muni = {}
        for feat in geo.get("features", []):
            p = feat.get("properties", {})
            muni = norm_loose(p.get("NAME_2") or p.get("city_name") or "")
            brgy = norm_loose(p.get("NAME_3") or p.get("brgy_name") or "")
            self.boundary_pairs.add((muni, brgy))
            self.boundary_by_muni.setdefault(muni, set()).add(brgy)

    def _build_hist_lookup(self):
        self.hist_lookup = {}
        grouped = self.raw_df.groupby(["Municipality","Barangay"], as_index=False)["DengueCases"].sum()
        grouped_2025 = self.raw_df.loc[self.raw_df["Year"] == 2025].groupby(["Municipality","Barangay"], as_index=False)["DengueCases"].sum()
        lookup_2025 = {(r["Municipality"], r["Barangay"]): int(r["DengueCases"]) for _, r in grouped_2025.iterrows()}
        for _, r in grouped.iterrows():
            muni_norm, brgy_norm = self._apply_overrides_to_pair(r["Municipality"], r["Barangay"])
            self.hist_lookup[(muni_norm, brgy_norm)] = {
                "historical_total_cases": int(r["DengueCases"]),
                "historical_2025_cases": int(lookup_2025.get((r["Municipality"], r["Barangay"]), 0)),
            }

    def _prepare_dataset(self, df: pd.DataFrame):
        req = ["Year","MorbidityWeek","Barangay","Municipality","DengueCases","RAINFALL_mm","TEMP_AVG_C","RELATIVE_HUMIDITY","Population"]
        df["Barangay"] = safe_upper_strip(df["Barangay"])
        df["Municipality"] = safe_upper_strip(df["Municipality"])
        for col in ["Year","MorbidityWeek","DengueCases","RAINFALL_mm","TEMP_AVG_C","RELATIVE_HUMIDITY","Population"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df.dropna(subset=req).copy()
        df["LocationKey"] = df["Municipality"] + " | " + df["Barangay"]
        positive_pop = df.loc[df["Population"] > 0].groupby("LocationKey")["Population"].median()
        bad = df["Population"] <= 0
        fillable = bad & df["LocationKey"].isin(positive_pop.index)
        df.loc[fillable, "Population"] = df.loc[fillable, "LocationKey"].map(positive_pop)
        df = df.loc[df["Population"] > 0].copy()
        raw = df.groupby(["Year","MorbidityWeek","Municipality","Barangay"], as_index=False).agg({"DengueCases":"sum","RAINFALL_mm":"mean","TEMP_AVG_C":"mean","RELATIVE_HUMIDITY":"mean","Population":"max"})
        raw["LocationKey"] = raw["Municipality"] + " | " + raw["Barangay"]
        raw = raw.sort_values(["LocationKey","Year","MorbidityWeek"]).reset_index(drop=True)
        model_df = raw.copy()
        model_df["CasesPer1000"] = (model_df["DengueCases"] / model_df["Population"]) * 1000.0
        model_df = make_week_cyclical(model_df)
        g = model_df.groupby("LocationKey", group_keys=False)
        for lag in [1,2,3,4,8,12]:
            model_df[f"cases_lag_{lag}"] = g["DengueCases"].shift(lag)
            model_df[f"rate_lag_{lag}"] = g["CasesPer1000"].shift(lag)
        for lag in [1,2,3,4]:
            model_df[f"rain_lag_{lag}"] = g["RAINFALL_mm"].shift(lag)
            model_df[f"temp_lag_{lag}"] = g["TEMP_AVG_C"].shift(lag)
            model_df[f"hum_lag_{lag}"] = g["RELATIVE_HUMIDITY"].shift(lag)
        for window in [2,4,8]:
            model_df[f"cases_rollsum_{window}"] = g["DengueCases"].transform(lambda s: s.shift(1).rolling(window=window, min_periods=1).sum())
            model_df[f"cases_rollmean_{window}"] = g["DengueCases"].transform(lambda s: s.shift(1).rolling(window=window, min_periods=1).mean())
            model_df[f"rate_rollmean_{window}"] = g["CasesPer1000"].transform(lambda s: s.shift(1).rolling(window=window, min_periods=1).mean())
            model_df[f"rate_rollmax_{window}"] = g["CasesPer1000"].transform(lambda s: s.shift(1).rolling(window=window, min_periods=1).max())
            model_df[f"rain_rollmean_{window}"] = g["RAINFALL_mm"].transform(lambda s: s.shift(1).rolling(window=window, min_periods=1).mean())
            model_df[f"temp_rollmean_{window}"] = g["TEMP_AVG_C"].transform(lambda s: s.shift(1).rolling(window=window, min_periods=1).mean())
            model_df[f"hum_rollmean_{window}"] = g["RELATIVE_HUMIDITY"].transform(lambda s: s.shift(1).rolling(window=window, min_periods=1).mean())
        model_df["loc_hist_mean_rate"] = g["CasesPer1000"].transform(lambda s: s.shift(1).expanding().mean())
        model_df["loc_hist_std_rate"] = g["CasesPer1000"].transform(lambda s: s.shift(1).expanding().std())
        model_df["loc_hist_max_rate"] = g["CasesPer1000"].transform(lambda s: s.shift(1).expanding().max())
        model_df["loc_hist_mean_cases"] = g["DengueCases"].transform(lambda s: s.shift(1).expanding().mean())
        model_df["loc_sameweek_mean_rate"] = add_expanding_sameweek_mean(model_df, ["LocationKey","MorbidityWeek"], "CasesPer1000")
        model_df["mun_sameweek_mean_rate"] = add_expanding_sameweek_mean(model_df, ["Municipality","MorbidityWeek"], "CasesPer1000")
        mun_week = model_df.groupby(["Municipality","Year","MorbidityWeek"], as_index=False).agg({"DengueCases":"sum","CasesPer1000":"mean","RAINFALL_mm":"mean"}).sort_values(["Municipality","Year","MorbidityWeek"]).reset_index(drop=True)
        g_mun = mun_week.groupby("Municipality", group_keys=False)
        mun_week["mun_cases_lag_1"] = g_mun["DengueCases"].shift(1)
        mun_week["mun_rate_lag_1"] = g_mun["CasesPer1000"].shift(1)
        mun_week["mun_rate_rollmean_4"] = g_mun["CasesPer1000"].transform(lambda s: s.shift(1).rolling(window=4, min_periods=1).mean())
        model_df = model_df.merge(mun_week[["Municipality","Year","MorbidityWeek","mun_cases_lag_1","mun_rate_lag_1","mun_rate_rollmean_4"]], on=["Municipality","Year","MorbidityWeek"], how="left")
        model_df["zero_streak"] = add_zero_streak(model_df, "LocationKey", "DengueCases")
        model_df["rain_x_humidity"] = model_df["RAINFALL_mm"] * model_df["RELATIVE_HUMIDITY"]
        model_df["temp_x_humidity"] = model_df["TEMP_AVG_C"] * model_df["RELATIVE_HUMIDITY"]
        model_df["cases_over_loc_mean"] = model_df["DengueCases"] / (model_df["loc_hist_mean_cases"] + 1.0)
        model_df["rate_over_loc_mean"] = model_df["CasesPer1000"] / (model_df["loc_hist_mean_rate"] + 1e-6)
        model_df = build_recent_surge_features(model_df)
        for c in self.feature_cols:
            if c not in model_df.columns:
                model_df[c] = 0.0
            model_df[c] = pd.to_numeric(model_df[c], errors="coerce").fillna(0.0)
        return model_df, raw

    def _next_week_from(self, year, week):
        return (year + 1, 1) if week >= 52 else (year, week + 1)

    def latest_available_target(self):
        row = self.model_df.sort_values(["Year","MorbidityWeek"]).iloc[-1]
        year, week = int(row["Year"]), int(row["MorbidityWeek"])
        for _ in range(FORECAST_HORIZON_WEEKS):
            year, week = self._next_week_from(year, week)
        return {"latest_available": {"year": int(row["Year"]), "week": int(row["MorbidityWeek"])}, "next_forecast": {"year": year, "week": week}, "forecast_horizon_weeks": FORECAST_HORIZON_WEEKS}

    def health(self):
        return {"dataset_loaded": True, "model_loaded": True, **self.latest_available_target()}

    def available_locations(self):
        out = self.model_df[["Municipality","Barangay"]].drop_duplicates().sort_values(["Municipality","Barangay"])
        return [{"municipality": r["Municipality"].title(), "barangay": r["Barangay"].title()} for _, r in out.iterrows()]

    def _apply_thresholds(self, probs):
        pred = int(np.argmax(probs))
        if probs[2] >= self.thr_high:
            return 2
        if probs[1] >= self.thr_low and probs[2] < self.thr_high:
            return 1
        return pred

    def _scale_features(self, features_df):
        arr = features_df[self.feature_cols].values.astype(np.float32)
        return ((arr - self.scaler_mean) / self.scaler_scale).astype(np.float32)

    def _predict_probs(self, engineered_loc_df):
        seq_source = engineered_loc_df.sort_values(["Year","MorbidityWeek"]).tail(SEQ_LEN).copy()
        scaled = self._scale_features(seq_source)
        x = torch.tensor(scaled[np.newaxis, :, :], dtype=torch.float32)
        with torch.no_grad():
            logits = self.model(x)
            return torch.softmax(logits, dim=1).cpu().numpy()[0]

    def _trend_label(self, series):
        if len(series) < 2:
            return "Insufficient data"
        return "Increasing" if series[-1] > series[-2] else ("Decreasing" if series[-1] < series[-2] else "Stable")

    def _apply_overrides_to_pair(self, municipality, barangay):
        muni_norm = {"BATANGAS":"BATANGASCITY", "LIPA":"LIPACITY", "TANAUAN":"TANAUANCITY"}.get(norm(municipality), norm(municipality))
        brgy_norm = norm_loose(barangay)
        candidates = [brgy_norm]
        if muni_norm == "BATANGASCITY" and brgy_norm in ISLA_VERDE_SPECIAL:
            candidates.append(ISLA_VERDE_SPECIAL[brgy_norm])
        if brgy_norm == "MAABUDNORTE":
            candidates.append("MAABUDNORTH")
        if brgy_norm == "MAABUDSUR":
            candidates.append("MAABUDSOUTH")
        m = re.match(r"^(.*?)(\d+)$", brgy_norm)
        if m:
            prefix, num = m.group(1), int(m.group(2))
            if num in ROMAN:
                candidates += [prefix + ROMAN[num], prefix + "BARANGAY" + str(num), prefix + "BARANGAY" + ROMAN[num]]
                if prefix == "POBLACION":
                    candidates += ["POBLACIONBARANGAY" + str(num), "POBLACIONBARANGAY" + ROMAN[num], "POBLACION" + ROMAN[num]]
        if brgy_norm.endswith("1"):
            candidates.append(brgy_norm[:-1])
        for c in candidates:
            if c in self.boundary_by_muni.get(muni_norm, set()):
                return muni_norm, c
        candidates_in_muni = list(self.boundary_by_muni.get(muni_norm, set()))
        if candidates_in_muni:
            best = max(candidates_in_muni, key=lambda c: difflib.SequenceMatcher(None, brgy_norm, c).ratio())
            if difflib.SequenceMatcher(None, brgy_norm, best).ratio() >= 0.92:
                return muni_norm, best
        return muni_norm, brgy_norm

    def _build_reason_summary(self, risk_label, probs, row, weekly_trend):
        no_risk_prob = float(probs[0])
        low_risk_prob = float(probs[1])
        high_risk_prob = float(probs[2])
        trend = self._trend_label(weekly_trend).lower()
        recent_cases = int(row["DengueCases"])
        population = int(row["Population"])
        rate = float(row["CasesPer1000"])
        if risk_label == HIGH_RISK_LABEL:
            return (
                f"This location is classified as HIGH_RISK because the model's high-risk probability ({high_risk_prob:.3f}) crossed the tuned high-risk decision threshold of {self.thr_high:.2f}. "
                f"That decision can still happen even when the most recent observed case count is {recent_cases}, because the model also weighs population ({population:,}), recent weather context, prior-week patterns, and historical risk behavior. "
                f"The latest short-term trend is {trend}, and the latest observed case rate is {rate:.4f} per 1,000 population."
            )
        if risk_label == LOW_RISK_LABEL:
            return (
                f"This location is classified as LOW_RISK because the low-risk probability ({low_risk_prob:.3f}) is strong enough under the tuned decision rules while high-risk probability stays below the high-risk cutoff. "
                f"Recent observed cases are {recent_cases}, population is {population:,}, and the short-term trend is {trend}."
            )
        return (
            f"This location is classified as NO_RISK because the no-risk probability ({no_risk_prob:.3f}) remains the strongest final signal and the high-risk probability ({high_risk_prob:.3f}) stays below the tuned high-risk threshold of {self.thr_high:.2f}. "
            f"Recent observed cases are {recent_cases}, population is {population:,}, and the short-term trend is {trend}."
        )

    def predict_one(self, municipality, barangay, horizon_weeks=2):
        loc_key = f"{municipality.strip().upper()} | {barangay.strip().upper()}"
        current_loc_df = self.model_df.loc[self.model_df["LocationKey"] == loc_key].sort_values(["Year","MorbidityWeek"]).copy()
        if current_loc_df.empty:
            raise ValueError(f"Location not found: {municipality} / {barangay}")
        probs = self._predict_probs(current_loc_df)
        pred_idx = self._apply_thresholds(probs)
        row = current_loc_df.iloc[-1]
        year, week = int(row["Year"]), int(row["MorbidityWeek"])
        for _ in range(horizon_weeks):
            year, week = self._next_week_from(year, week)
        weekly_trend = current_loc_df["DengueCases"].tail(14).astype(int).tolist()
        muni_norm, brgy_norm = self._apply_overrides_to_pair(row["Municipality"], row["Barangay"])
        risk_label = INT_TO_RISK[pred_idx]
        return {
            "municipality": str(row["Municipality"]).title(),
            "barangay": str(row["Barangay"]).title(),
            "municipality_norm": muni_norm,
            "barangay_norm": brgy_norm,
            "forecast_year": year,
            "forecast_week": week,
            "context_year": int(row["Year"]),
            "context_week": int(row["MorbidityWeek"]),
            "risk_label": risk_label,
            "prediction_score": round(float(probs[pred_idx]), 6),
            "probabilities": {NO_RISK_LABEL: round(float(probs[0]), 6), LOW_RISK_LABEL: round(float(probs[1]), 6), HIGH_RISK_LABEL: round(float(probs[2]), 6)},
            "supporting_indicators": {"rainfall_mm": round(float(row["RAINFALL_mm"]), 3), "temperature_c": round(float(row["TEMP_AVG_C"]), 3), "humidity_percent": round(float(row["RELATIVE_HUMIDITY"]), 3), "population": int(row["Population"]), "cases_per_1000": round(float(row["CasesPer1000"]), 6)},
            "trend": self._trend_label(weekly_trend),
            "last_observed_cases": int(row["DengueCases"]),
            "weekly_trend": weekly_trend,
            "historical_total_cases": self.hist_lookup.get((muni_norm, brgy_norm), {}).get("historical_total_cases"),
            "historical_2025_cases": self.hist_lookup.get((muni_norm, brgy_norm), {}).get("historical_2025_cases"),
            "reason_summary": self._build_reason_summary(risk_label, probs, row, weekly_trend),
            "notes": "Updated optimized best 2-week LSTM artifacts loaded and used for this forecast.",
        }

    def predict_all(self):
        if self._predict_cache is not None:
            return self._predict_cache
        items = []
        for _, row in self.model_df[["Municipality","Barangay"]].drop_duplicates().iterrows():
            try:
                items.append(self.predict_one(row["Municipality"], row["Barangay"], 2))
            except Exception:
                pass
        self._predict_cache = items
        return items

    def _municipality_weekly_trend(self, municipality: str):
        sub = self.raw_df.loc[self.raw_df["Municipality"] == municipality.upper()].sort_values(["Year","MorbidityWeek"])
        if sub.empty:
            return []
        return sub.groupby(["Year","MorbidityWeek"], as_index=False)["DengueCases"].sum().tail(14)["DengueCases"].astype(int).tolist()

    def polygon_heatmap_counts(self):
        items = self.predict_all()
        exact = {(x["municipality_norm"], x["barangay_norm"]): x for x in items}
        muni_roll = {}
        for x in items:
            k = x["municipality_norm"]
            cur = muni_roll.get(k)
            if (cur is None) or (x["risk_label"] == HIGH_RISK_LABEL) or (x["risk_label"] == LOW_RISK_LABEL and cur["risk_label"] == NO_RISK_LABEL):
                muni_roll[k] = x
        counts = {HIGH_RISK_LABEL: 0, LOW_RISK_LABEL: 0, NO_RISK_LABEL: 0, "NO_DATA": 0}
        for muni, brgy in self.boundary_pairs:
            item = exact.get((muni, brgy)) or muni_roll.get(muni)
            counts[item["risk_label"] if item else "NO_DATA"] += 1
        return counts

    def map_geojson(self):
        boundary_path = self.project_root.parent / "frontend" / "public" / "batangas_barangays.geojson"
        with open(boundary_path, "r", encoding="utf-8") as f:
            geo = json.load(f)
        items = self.predict_all()
        exact = {(x["municipality_norm"], x["barangay_norm"]): x for x in items}
        muni_roll = {}
        for x in items:
            k = x["municipality_norm"]
            cur = muni_roll.get(k)
            if (cur is None) or (x["risk_label"] == HIGH_RISK_LABEL) or (x["risk_label"] == LOW_RISK_LABEL and cur["risk_label"] == NO_RISK_LABEL):
                muni_roll[k] = x
        out = {"type": "FeatureCollection", "features": []}
        for feat in geo.get("features", []):
            p = dict(feat.get("properties", {}))
            muni_name = p.get("NAME_2") or p.get("city_name") or ""
            brgy_name = p.get("NAME_3") or p.get("brgy_name") or ""
            muni_key = norm_loose(muni_name)
            brgy_key = norm_loose(brgy_name)
            exact_item = exact.get((muni_key, brgy_key))
            muni_item = muni_roll.get(muni_key)
            source = "exact" if exact_item else ("municipality" if muni_item else "none")
            chosen = exact_item or muni_item
            p["municipality_name"] = str(muni_name)
            p["barangay_name"] = str(brgy_name)
            p["municipality_norm"] = muni_key
            p["barangay_norm"] = brgy_key
            p["match_source"] = source
            hist_info = self.hist_lookup.get((muni_key, brgy_key), {})
            p["historical_total_cases"] = hist_info.get("historical_total_cases")
            p["historical_2025_cases"] = hist_info.get("historical_2025_cases")
            if chosen:
                p["risk_label"] = chosen["risk_label"]
                p["prediction_score"] = chosen["prediction_score"]
                p["weekly_trend"] = chosen["weekly_trend"] if exact_item else self._municipality_weekly_trend(muni_name)
                p["context_year"] = chosen.get("context_year")
                p["context_week"] = chosen.get("context_week")
                p["forecast_year"] = chosen.get("forecast_year")
                p["forecast_week"] = chosen.get("forecast_week")
                p["last_observed_cases"] = chosen.get("last_observed_cases")
                p["reason_summary"] = chosen.get("reason_summary") if exact_item else "Municipality-level fallback was used because this polygon has no exact barangay forecast match."
                p["supporting_indicators"] = chosen.get("supporting_indicators", {})
                p["info_note"] = "Exact barangay forecast match." if exact_item else "Municipality-level fallback used because this polygon has no exact barangay forecast match."
            else:
                p["risk_label"] = "NO_DATA"
                p["prediction_score"] = None
                p["weekly_trend"] = []
                p["last_observed_cases"] = None
                p["reason_summary"] = "No matched forecast found for this polygon."
                p["supporting_indicators"] = {}
                p["info_note"] = "No matched forecast found for this polygon."
            out["features"].append({"type": "Feature", "properties": p, "geometry": feat.get("geometry")})
        return out

    def match_report(self):
        if self._match_cache is not None:
            return self._match_cache
        matched = 0
        unmatched = []
        for _, row in self.model_df[["Municipality","Barangay"]].drop_duplicates().iterrows():
            k = self._apply_overrides_to_pair(row["Municipality"], row["Barangay"])
            if k in self.boundary_pairs:
                matched += 1
            else:
                unmatched.append({"municipality": str(row["Municipality"]).title(), "barangay": str(row["Barangay"]).title()})
        self._match_cache = {"matched_count": matched, "unmatched_count": len(unmatched), "sample_unmatched": unmatched[:20]}
        return self._match_cache

    def summary(self):
        if self._summary_cache is not None:
            return self._summary_cache
        items = self.predict_all()
        polygon_counts = self.polygon_heatmap_counts()
        hist = self.raw_df.copy()
        top_hist_all = hist.groupby("Municipality", as_index=False)["DengueCases"].sum().sort_values("DengueCases", ascending=False).head(10)
        top_brgy_all = hist.groupby(["Municipality","Barangay"], as_index=False)["DengueCases"].sum().sort_values("DengueCases", ascending=False).head(10)
        years = sorted(hist["Year"].dropna().astype(int).unique().tolist())
        by_year = {}
        for year in years:
            sub = hist.loc[hist["Year"] == year].copy()
            top_m = sub.groupby("Municipality", as_index=False)["DengueCases"].sum().sort_values("DengueCases", ascending=False).head(10)
            top_b = sub.groupby(["Municipality","Barangay"], as_index=False)["DengueCases"].sum().sort_values("DengueCases", ascending=False).head(10)
            by_year[str(year)] = {
                "top_municipalities": [{"municipality": r["Municipality"].title(), "cases": int(r["DengueCases"])} for _, r in top_m.iterrows()],
                "top_barangays": [{"barangay": f"{r['Barangay'].title()}, {r['Municipality'].title()}", "cases": int(r["DengueCases"])} for _, r in top_b.iterrows()],
            }
        muni_summary = {}
        for item in items:
            muni = item["municipality"]
            e = muni_summary.setdefault(muni, {"HIGH_RISK":0, "LOW_RISK":0, "NO_RISK":0, "avg_score":[]})
            e[item["risk_label"]] += 1
            e["avg_score"].append(item["prediction_score"])
        top = [{"municipality": m, "high": v["HIGH_RISK"], "low": v["LOW_RISK"], "no_risk": v["NO_RISK"], "avg_score": round(float(np.mean(v["avg_score"])), 4)} for m, v in muni_summary.items()]
        top = sorted(top, key=lambda x: (x["high"], x["avg_score"]), reverse=True)[:10]
        self._summary_cache = {
            "risk_counts": {"HIGH_RISK": polygon_counts[HIGH_RISK_LABEL], "LOW_RISK": polygon_counts[LOW_RISK_LABEL], "NO_RISK": polygon_counts[NO_RISK_LABEL]},
            "polygon_total": sum(polygon_counts.values()),
            "matched_pairs": self.match_report()["matched_count"],
            "unmatched_pairs": self.match_report()["unmatched_count"],
            "top_municipalities": top,
            "historical_top_municipalities": [{"municipality": r["Municipality"].title(), "cases": int(r["DengueCases"])} for _, r in top_hist_all.iterrows()],
            "historical_top_barangays": [{"barangay": f"{r['Barangay'].title()}, {r['Municipality'].title()}", "cases": int(r["DengueCases"])} for _, r in top_brgy_all.iterrows()],
            "historical_by_year": by_year,
            "years": years,
            "historical_stats": {"total_cases": int(hist["DengueCases"].sum()), "barangays": int(hist[["Municipality","Barangay"]].drop_duplicates().shape[0]), "municipalities": int(hist["Municipality"].nunique()), "year_start": int(hist["Year"].min()), "year_end": int(hist["Year"].max())},
            "forecast_target": self.latest_available_target()["next_forecast"],
        }
        return self._summary_cache

@lru_cache(maxsize=1)
def get_service():
    return ForecastService(Path(__file__).resolve().parents[1])
