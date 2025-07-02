import pandas as pd
from src.data_loader import (
    load_last_500_rows, load_new_booking, append_to_master, ensure_master_files
)
from src.occupancy_model import train_and_save as train_occ, predict as predict_occ, make_occ_features
from src.rewards_model   import train_and_save as train_rew,  predict as predict_rew
from src.pricing_model   import train_and_save as train_price,predict as predict_price, make_pricing_features

ensure_master_files()

occ_df      = load_last_500_rows("data/occupancy_dataset.xlsx")
pricing_df  = load_last_500_rows("data/pricing_dataset.xlsx")
new_booking = load_new_booking  ("data/new_booking.xlsx")

occ_model,  occ_metrics  = train_occ (occ_df)
rew_model,  rew_metrics  = train_rew (pricing_df)
price_model,price_metrics= train_price(pricing_df)

occ_cls = predict_occ (occ_model,  new_booking)
rew_cls = predict_rew (rew_model,  new_booking)
price   = predict_price(price_model,new_booking)

# build engineered rows to append
occ_row = make_occ_features(new_booking)
occ_row["occ_class"] = occ_cls
occ_row = occ_row.reindex(occ_df.columns.tolist()+["occ_class"], axis=1, fill_value=pd.NA)
append_to_master("data/occupancy_dataset.xlsx", occ_row)

price_row = make_pricing_features(new_booking)
price_row["points_class"] = rew_cls
price_row["final_price"]  = price
price_row = price_row.reindex(pricing_df.columns.tolist()+["points_class","final_price"],
                              axis=1, fill_value=pd.NA)
append_to_master("data/pricing_dataset.xlsx", price_row)

print("✅ Occupancy Class:", occ_cls)
print("✅ Points Class:",    rew_cls)
print("✅ Final Room Price: $", round(price, 2))
print("📊 Occupancy Metrics:", occ_metrics)
print("📊 Rewards Accuracy:",  rew_metrics["accuracy"])
print("📊 Pricing Metrics:",   price_metrics)
