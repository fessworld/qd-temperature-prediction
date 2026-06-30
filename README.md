# Quantum Dot Temperature Prediction with Python and Dask

This repository contains a reproducible Python workflow for predicting the measurement temperature of single perovskite quantum dots from optical features extracted from emission spectra and photoluminescence decay kinetics.

The project was developed for the microcredential **Aplicación práctica avanzada de Python para investigaciones científicas con Big Data** and uses a real research workflow: experimental files are transformed into a feature table, then regression models are trained and evaluated with leakage-aware group splits.

## Scientific question

Can the temperature of an individual quantum dot be estimated from optical properties such as:

- emission peak energy,
- spectral linewidth, FWHM,
- photoluminescence lifetime descriptors?

The final model is a `RandomForestRegressor` trained on engineered spectral and lifetime features. Dask is used for scalable data loading and for parallel hyperparameter search with `GridSearchCV`.

## Repository structure

```text
qd-temperature-prediction/
├── data/
│   └── README.md                       # Data policy and expected schema
├── docs/
│   └── methodology.md                  # Scientific and technical method summary
├── notebooks/
│   └── README.md                       # Where to place notebooks
├── scripts/
│   ├── make_synthetic_demo_data.py     # Creates a public demo dataset
│   └── train_temperature_model.py      # End-to-end model training script
├── src/qdtemp/
│   ├── __init__.py
│   ├── model.py                        # Data preparation, training, Dask grid search
│   └── visualization.py                # Plot helpers
├── tests/
│   └── test_model.py                   # Minimal tests for public code
├── requirements.txt
├── environment.yml
├── pyproject.toml
├── .gitignore
├── LICENSE
└── CITATION.cff
```

## Data privacy

The original experimental data are not included because they may contain lab-specific paths, unpublished measurements and internal metadata. The repository expects a feature table with the same schema as `features_with_tau_real.csv`. A synthetic demo dataset can be generated to test the code.

Expected columns:

```text
qd_id, temperature_K, peak_eV, fwhm_nm,
tau_moment_ns_raw, tau_ampw_ns_raw, tau_intw_ns_raw
```

Optional columns from the full internal workflow, such as reconvolved lifetimes or IRF metadata, can be added later.

## Quick start

### 1. Create environment

With conda:

```bash
conda env create -f environment.yml
conda activate qd-temp-ml
```

Or with pip:

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
```

### 2. Generate demo data

```bash
python scripts/make_synthetic_demo_data.py --out data/features_with_tau_real.csv
```

### 3. Train models

```bash
python scripts/train_temperature_model.py \
  --input data/features_with_tau_real.csv \
  --output outputs/temp_prediction \
  --use-dask-grid
```

Outputs include:

- `metrics_baseline.csv`
- `metrics_rf_grid_dask.csv`
- `feature_importances_random_forest.csv`
- `pred_vs_true_*.png`
- `feature_importances_*.png`

## Model overview

The workflow trains and compares:

1. `DummyRegressor`, constant mean baseline.
2. `Ridge`, standardized linear model.
3. `RandomForestRegressor`, nonlinear model.
4. Optional Dask-backed `GridSearchCV` for Random Forest hyperparameter tuning.

The split is grouped by `qd_id`, so one quantum dot cannot appear in both training and test sets. This avoids leakage from dot-specific behavior.

## Reproducibility notes

- Random seeds are fixed.
- File paths are command-line arguments.
- Outputs are saved to a dedicated folder.
- Environment files are included.
- Minimal tests are included.

## License

The code is released under the MIT License. Do not upload unpublished raw data unless your lab or institution explicitly allows it.
