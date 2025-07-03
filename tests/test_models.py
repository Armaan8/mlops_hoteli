# tests/test_models.py
import importlib
import pytest
from src.data_loader import load_last_500_rows, load_new_booking

@pytest.mark.parametrize("module_name, master_file", [
    ("occupancy_model", "data/occupancy_dataset.xlsx"),
    ("rewards_model",   "data/pricing_dataset.xlsx"),
    ("pricing_model",   "data/pricing_dataset.xlsx"),
])
def test_model_contract(module_name, master_file):
    # Dynamically import the module
    module = importlib.import_module(f"src.{module_name}")
    # 1) Must have train_and_save and predict
    assert hasattr(module, "train_and_save"), f"{module_name} missing train_and_save"
    assert hasattr(module, "predict"),       f"{module_name} missing predict"
    # 2) train_and_save returns (model, metrics dict)
    df = load_last_500_rows(master_file)
    model, metrics = module.train_and_save(df)
    assert isinstance(metrics, dict), "Metrics must be a dict"
    # 3) predict returns something non-null on a real booking
    booking = load_new_booking("data/new_booking.xlsx")
    pred = module.predict(model, booking)
    assert pred is not None, "Predict returned None"
