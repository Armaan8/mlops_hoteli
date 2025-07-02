# pipeline_runner.py
from pathlib import Path
import pandas as pd

from src.data_loader import (
    load_last_500_rows,
    load_new_booking,
    append_to_master,
    ensure_master_files,
)

from src.occupancy_model import (
    train_and_save as train_occ,
    predict        as predict_occ,
    make_occ_features,
)
from src.rewards_model  import (
    train_and_save as train_rew,
    predict        as predict_rew,
)
from src.pricing_model  import (
    train_and_save as train_price,
    predict        as predict_price,
    make_pricing_features,
)

# ───────────────────────────────────────────────────────────────
# 0️⃣  guarantee base folders + empty master files exist
ensure_master_files()

# booking file path
BOOKING_PATH = Path("data/new_booking.xlsx")
has_booking  = BOOKING_PATH.exists()

# ───────────────────────────────────────────────────────────────
# 1️⃣  load master datasets (last 500 rows each)
occ_df     = load_last_500_rows("data/occupancy_dataset.xlsx")
pricing_df = load_last_500_rows("data/pricing_dataset.xlsx")

# ───────────────────────────────────────────────────────────────
# 2️⃣  always retrain (training-only run is still useful nightly / in CI)
occ_model,   occ_metrics   = train_occ(occ_df)
rew_model,   rew_metrics   = train_rew(pricing_df)
price_model, price_metrics = train_price(pricing_df)

# ───────────────────────────────────────────────────────────────
# 3️⃣  predict & append **only** when a booking file is present
if has_booking:
    new_booking = load_new_booking(str(BOOKING_PATH))

    # ─ predictions ─
    occ_cls = predict_occ  (occ_model,   new_booking)
    rew_cls = predict_rew  (rew_model,   new_booking)
    price   = predict_price(price_model, new_booking)

    # ─ append engineered occupancy row ─
    occ_row = make_occ_features(new_booking)
    occ_row["occ_class"] = occ_cls
    occ_row = occ_row.reindex(
        occ_df.columns.tolist() + ["occ_class"], axis=1, fill_value=pd.NA
    )
    append_to_master("data/occupancy_dataset.xlsx", occ_row)

    # ─ append engineered pricing row ─
    price_row = make_pricing_features(new_booking)
    price_row["points_class"] = rew_cls
    price_row["final_price"]  = price
    price_row = price_row.reindex(
        pricing_df.columns.tolist() + ["points_class", "final_price"],
        axis=1, fill_value=pd.NA,
    )
    append_to_master("data/pricing_dataset.xlsx", price_row)

    # ─ console summary ─
    print("✅ Occupancy Class:", occ_cls)
    print("✅ Points Class:   ", rew_cls)
    print("✅ Final Room Price: $", round(price, 2))
else:
    print("📭  No new_booking.xlsx found — ran training only.")

# always show metrics
print("📊 Occupancy Metrics:", occ_metrics)
print("📊 Rewards Accuracy: ", rew_metrics["accuracy"])
print("📊 Pricing Metrics:  ", price_metrics)
print("✅ Pipeline Execution Completed.")
