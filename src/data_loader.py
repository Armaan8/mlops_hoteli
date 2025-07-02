import warnings
warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    module="pandas",
    message="The behavior of DataFrame concatenation",
)

import pandas as pd
from pathlib import Path

OCCUPANCY_FILE = Path("data/occupancy_dataset.xlsx")
PRICING_FILE   = Path("data/pricing_dataset.xlsx")
NEW_BOOK_PATH  = Path("data/new_booking.xlsx")


# ───────────────────────── helpers ──────────────────────────
def load_last_500_rows(path: str | Path) -> pd.DataFrame:
    return pd.read_excel(path).tail(500).copy()


def load_new_booking(path: str | Path) -> pd.DataFrame:
    return pd.read_excel(path)


def _dedup_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Force column labels unique (adds _1, _2 …)."""
    cols = pd.Series(df.columns)
    dup  = cols.duplicated()
    for idx in cols[dup].index:
        base = cols[idx]
        k = 1
        while f"{base}_{k}" in cols.values:
            k += 1
        cols[idx] = f"{base}_{k}"
    df.columns = cols
    return df


def append_to_master(master_path: str | Path, new_row: pd.DataFrame) -> None:
    try:
        master = pd.read_excel(master_path)
    except Exception:
        master = pd.DataFrame()

    master  = _dedup_columns(master)
    new_row = _dedup_columns(new_row)

    new_row = new_row.reindex(master.columns, axis=1, fill_value=pd.NA)
    if master.empty and master.columns.empty:
        master = new_row
    else:
        master = pd.concat([master, new_row], ignore_index=True)

    master.to_excel(master_path, index=False)


def ensure_master_files() -> None:
    for fp in (OCCUPANCY_FILE, PRICING_FILE):
        if not fp.exists():
            pd.DataFrame().to_excel(fp, index=False)
