import pandas as pd
import numpy as np
import os, joblib
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# ─────────────────────────────────────────────────────────────
MODEL_DIR     = "models/pricing"
FEATURE_PATH  = os.path.join(MODEL_DIR, "feature_columns.npy")
MEAN_PATH     = os.path.join(MODEL_DIR, "feature_means.npy")   # NEW
os.makedirs(MODEL_DIR, exist_ok=True)

TARGET_COL = "final_bfr_usd"           # numeric target

# ─────────────────────────────────────────────────────────────
def make_pricing_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "points_class" in df.columns:
        df["points_class"] = LabelEncoder().fit_transform(
            df["points_class"].astype(str)
        )
    df = pd.get_dummies(df, drop_first=True)
    return df

def _numeric(df: pd.DataFrame) -> pd.DataFrame:
    return df.select_dtypes(include=[np.number])

# ─────────────────────────────────────────────────────────────
def preprocess(df: pd.DataFrame, *, return_feature_names=False):
    if TARGET_COL not in df.columns:
        raise ValueError(f"{TARGET_COL} not found in pricing dataset.")

    df = df.dropna(subset=[TARGET_COL])               # ensure target not NaN
    if df[TARGET_COL].nunique() < 2:
        raise ValueError(f"{TARGET_COL} needs ≥2 distinct values.")

    df = df.copy()
    y  = df[TARGET_COL]
    df = df.drop(columns=[TARGET_COL])

    if "points_class" in df.columns:
        df["points_class"] = LabelEncoder().fit_transform(
            df["points_class"].astype(str)
        )

    df   = pd.get_dummies(df, drop_first=True)
    X_df = _numeric(df)               # keep numeric predictors
    feature_means = X_df.mean().to_numpy()   # NEW: store column means
    X     = SimpleImputer(strategy="mean").fit_transform(X_df)

    if return_feature_names:
        return X, y, X_df.columns.to_numpy(), feature_means
    return X, y

# ─────────────────────────────────────────────────────────────
def train_and_save(df: pd.DataFrame):
    X, y, feat_cols, feat_means = preprocess(df, return_feature_names=True)

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = ElasticNet(random_state=42, positive=True)
    model.fit(X_tr, y_tr)

    y_pred = model.predict(X_te)
    metrics = {
        "target_used": TARGET_COL,
        "RMSE": round(np.sqrt(mean_squared_error(y_te, y_pred)), 4),
        "MAE" : round(mean_absolute_error(y_te, y_pred),        4),
        "R2"  : round(r2_score(y_te, y_pred),                  4),
    }
    
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    joblib.dump(model,  f"{MODEL_DIR}/model_{ts}.pkl")
    joblib.dump(model,  f"{MODEL_DIR}/latest.pkl")
    np.save(FEATURE_PATH, feat_cols)
    np.save(MEAN_PATH,     feat_means)          # NEW: save means

    return model, metrics

# ─────────────────────────────────────────────────────────────
def predict(model, booking_df: pd.DataFrame) -> float:
    df = make_pricing_features(booking_df)

    # align columns; keep NaN for missing so we can fill with means
    feat_cols  = np.load(FEATURE_PATH, allow_pickle=True)
    feat_means = np.load(MEAN_PATH,    allow_pickle=True)
    df = df.reindex(feat_cols, axis=1, fill_value=np.nan)
    df = df.apply(pd.to_numeric, errors="coerce")  # ensure float

    # replace NaN with training means (per column)
    X = df.to_numpy(dtype=float)
    nan_mask = np.isnan(X)
    X[nan_mask] = np.take(feat_means, np.where(nan_mask)[1])

    return float(model.predict(X)[0])      # no artificial floor
