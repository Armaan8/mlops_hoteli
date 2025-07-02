import pandas as pd
import numpy as np
import os, joblib
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, accuracy_score

# ─────────────────────────────────────────────────────────────
MODEL_DIR    = "models/rewards"
FEATURE_PATH = os.path.join(MODEL_DIR, "feature_columns.npy")
os.makedirs(MODEL_DIR, exist_ok=True)


# ─────────────────────────────────────────────────────────────
def create_points_class(row):
    """Pick the class with highest probability."""
    probs   = [row["prob_L"], row["prob_M"], row["prob_H"]]
    classes = ["L", "M", "H"]
    return classes[np.argmax(probs)]


def preprocess_rewards(df: pd.DataFrame, *, return_feature_names=False):
    df = df.copy()
    df["points_class"] = df.apply(create_points_class, axis=1)

    # remove columns you do NOT want for training
    drop_cols = [
        "mnemonic_cd", "PROP_NM", "BRAND_CD", "BRAND_NM", "CHAIN_CD", "CHAIN_NM",
        "CITY_NM", "market_name", "curr_cd", "previous_BFR_type", "stay_date",
    ]
    df = df.drop(columns=drop_cols, errors="ignore")

    # one-hot encode categoricals
    df = pd.get_dummies(
        df,
        columns=[
            "BRAND_CATEGORY",
            "region",
            "location_type",
            "BFR_type",
            "dow",
            "Top_Market_Flag",
        ],
        drop_first=True,
    )

    df = df.dropna(subset=["points_class"])

    # ️⃣  FEATURES / TARGET
    X_df = df.drop(columns=["points_class"])
    y    = LabelEncoder().fit_transform(df["points_class"])

    # keep numeric only → avoids “string to float” crash
    X_df = X_df.select_dtypes(include=[np.number])

    # impute remaining NaNs
    X = SimpleImputer(strategy="mean").fit_transform(X_df)

    if return_feature_names:
        return X, y, X_df.columns.to_numpy()
    return X, y


# ─────────────────────────────────────────────────────────────
def train_and_save(df: pd.DataFrame):
    X, y, feat_cols = preprocess_rewards(df, return_feature_names=True)

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_tr, y_tr)

    y_pred   = model.predict(X_te)
    accuracy = round(accuracy_score(y_te, y_pred), 4)
    report   = classification_report(y_te, y_pred, output_dict=True)

    # persist model + feature list
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    joblib.dump(model, f"{MODEL_DIR}/model_{ts}.pkl")
    joblib.dump(model, f"{MODEL_DIR}/latest.pkl")
    np.save(FEATURE_PATH, feat_cols)

    return model, {"accuracy": accuracy, "report": report}


# ─────────────────────────────────────────────────────────────
def predict(model, booking_df: pd.DataFrame) -> str:
    df = booking_df.copy()

    df = pd.get_dummies(
        df,
        columns=[
            "BRAND_CATEGORY",
            "region",
            "location_type",
            "BFR_type",
            "dow",
            "Top_Market_Flag",
        ],
        drop_first=True,
    )

    # align to training column order
    feat_cols = np.load(FEATURE_PATH, allow_pickle=True)
    df = df.reindex(feat_cols, axis=1, fill_value=0)

    X_new = SimpleImputer(strategy="mean").fit_transform(df)
    pred_class = model.predict(X_new)[0]
    return ["L", "M", "H"][pred_class]
