from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from joblib import parallel_backend

FEATURE_COLUMNS = [
    "peak_eV",
    "fwhm_nm",
    "tau_moment_ns_raw",
    "tau_ampw_ns_raw",
    "tau_intw_ns_raw",
]

REQUIRED_COLUMNS = ["qd_id", "temperature_K", *FEATURE_COLUMNS]


@dataclass
class SplitData:
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    groups_train: pd.Series
    groups_test: pd.Series


def load_feature_table(path: str | Path) -> pd.DataFrame:
    """Load the feature table from CSV."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Feature table not found: {path}")
    return pd.read_csv(path)


def prepare_modeling_table(
    df: pd.DataFrame,
    feature_columns: Iterable[str] = FEATURE_COLUMNS,
    aggregate: bool = True,
) -> tuple[pd.DataFrame, list[str]]:
    """Prepare a clean modeling table.

    The function keeps only available required columns, converts numeric columns,
    optionally aggregates repeated spectra by (qd_id, temperature_K), and drops
    rows with missing target or feature values.
    """
    feature_columns = [c for c in feature_columns if c in df.columns]
    required = ["qd_id", "temperature_K", *feature_columns]
    missing_core = [c for c in ["qd_id", "temperature_K"] if c not in df.columns]
    if missing_core:
        raise ValueError(f"Missing core columns: {missing_core}")

    d = df[required].copy()
    d["qd_id"] = pd.to_numeric(d["qd_id"], errors="coerce").astype("Int64")
    for col in [c for c in d.columns if c != "qd_id"]:
        d[col] = pd.to_numeric(d[col], errors="coerce")

    d = d.dropna(subset=["qd_id", "temperature_K"])
    d["qd_id"] = d["qd_id"].astype(int)

    if aggregate:
        group_keys = ["qd_id", "temperature_K"]
        agg_cols = [c for c in d.columns if c not in group_keys]
        d = d.groupby(group_keys, as_index=False)[agg_cols].median(numeric_only=True)

    d = d.dropna(subset=["temperature_K", *feature_columns]).copy()
    return d, feature_columns


def group_split(
    d_model: pd.DataFrame,
    feature_columns: list[str],
    test_size: float = 0.2,
    random_state: int = 42,
) -> SplitData:
    """Group-aware train/test split using qd_id as the group label."""
    X = d_model[feature_columns]
    y = d_model["temperature_K"]
    groups = d_model["qd_id"]

    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_idx, test_idx = next(splitter.split(X, y, groups))

    return SplitData(
        X_train=X.iloc[train_idx],
        X_test=X.iloc[test_idx],
        y_train=y.iloc[train_idx],
        y_test=y.iloc[test_idx],
        groups_train=groups.iloc[train_idx],
        groups_test=groups.iloc[test_idx],
    )


def make_models(feature_columns: list[str], random_state: int = 42) -> dict[str, object]:
    """Create baseline models."""
    preproc = ColumnTransformer(
        [("num", StandardScaler(), feature_columns)],
        remainder="drop",
    )
    return {
        "DummyMean": DummyRegressor(strategy="mean"),
        "Ridge": Pipeline([
            ("prep", preproc),
            ("mdl", Ridge(alpha=1.0, random_state=random_state)),
        ]),
        "RandomForest": RandomForestRegressor(
            n_estimators=400,
            max_depth=None,
            min_samples_leaf=2,
            random_state=random_state,
            n_jobs=-1,
        ),
    }


def evaluate_predictions(y_true: pd.Series, y_pred: np.ndarray) -> dict[str, float]:
    """Return MAE, RMSE and R2 for regression."""
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    return {
        "MAE_K": float(mae),
        "RMSE_K": float(mse ** 0.5),
        "R2": float(r2_score(y_true, y_pred)),
    }


def run_dask_grid_search(
    split: SplitData,
    random_state: int = 42,
    param_grid: dict | None = None,
) -> GridSearchCV:
    """Run a GridSearchCV for RandomForest with a Dask joblib backend.

    A Dask client must already be active before calling this function.
    """
    if param_grid is None:
        param_grid = {
            "n_estimators": [200, 400, 800],
            "max_depth": [None, 8, 16],
            "min_samples_leaf": [1, 2, 4],
        }

    model = RandomForestRegressor(random_state=random_state, n_jobs=-1)
    cv = GroupKFold(n_splits=4)
    grid = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=cv,
        scoring="neg_mean_absolute_error",
        n_jobs=-1,
        verbose=1,
        refit=True,
    )
    with parallel_backend("dask"):
        grid.fit(split.X_train, split.y_train, groups=split.groups_train)
    return grid
