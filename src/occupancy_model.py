import pandas as pd, numpy as np, os, joblib
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

MODEL_DIR = "models/occupancy"
FEATURE_PATH = os.path.join(MODEL_DIR, "feature_columns.npy")
os.makedirs(MODEL_DIR, exist_ok=True)


def _base_feats(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["actv_dt"] = pd.to_datetime(df["actv_dt"])
    df["stay_date"] = pd.to_datetime(df["stay_date"])
    df["stay_dayofweek"] = df["stay_date"].dt.dayofweek
    df["stay_month"] = df["stay_date"].dt.month
    df["booking_lead_days"] = (df["stay_date"] - df["actv_dt"]).dt.days

    for c in ["CHAIN_CD", "BRAND_CD", "CITY_NM", "region"]:
        df[c] = LabelEncoder().fit_transform(df[c].astype(str))

    drop = [
        "mnemonic_cd","PROP_NM","CHAIN_NM","BRAND_NM","CTRY_NM","BFR_type",
        "actv_dt","stay_date","market_name","pred_occ_rf","pred_occ_class_rf",
    ]
    df = df.drop(columns=[c for c in drop if c in df.columns], errors="ignore")

    for c in df.select_dtypes(include="object").columns:
        df[c] = LabelEncoder().fit_transform(df[c].astype(str))
    return df


def make_occ_features(df: pd.DataFrame) -> pd.DataFrame:
    """Used by pipeline to build row with all engineered cols."""
    return _base_feats(df)


def _preprocess_train(df: pd.DataFrame):
    df = _base_feats(df).dropna(subset=["occ_ason_actv_dt"])
    X_df = df.drop(columns=["occ_ason_actv_dt"])
    y = df["occ_ason_actv_dt"]
    X = SimpleImputer(strategy="mean").fit_transform(X_df)
    return X, y, X_df.columns.to_numpy()


def train_and_save(df: pd.DataFrame):
    X, y, cols = _preprocess_train(df)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)

    model = GradientBoostingRegressor(n_estimators=100, random_state=42)
    model.fit(Xtr, ytr)

    np.save(FEATURE_PATH, cols)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    joblib.dump(model, f"{MODEL_DIR}/model_{ts}.pkl")
    joblib.dump(model, f"{MODEL_DIR}/latest.pkl")

    yp = model.predict(Xte)
    return model, {
        "RMSE": round(np.sqrt(mean_squared_error(yte, yp)), 4),
        "MAE":  round(mean_absolute_error(yte, yp), 4),
        "R2":   round(r2_score(yte, yp), 4),
    }


def predict(model, booking_df: pd.DataFrame) -> str:
    df = _base_feats(booking_df)
    cols = np.load(FEATURE_PATH, allow_pickle=True)
    df = df.reindex(cols, axis=1, fill_value=0)
    X = SimpleImputer(strategy="mean").fit_transform(df)
    s = model.predict(X)[0]
    return "High" if s > 0.7 else "Medium" if s > 0.4 else "Low"
