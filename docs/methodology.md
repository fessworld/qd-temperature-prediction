# Methodology

## Feature table

The modeling table is built from spectra and photoluminescence decay measurements. In the internal research workflow, each spectral file is processed by detecting the main emission peak and fitting a Lorentzian profile. This produces peak position, FWHM and amplitude above background. Each decay file is summarized by lifetime descriptors from moment and bi-exponential analysis.

## Target

The supervised target is `temperature_K`, the measurement temperature in kelvin.

## Features

The public model uses:

- `peak_eV`, emission peak energy,
- `fwhm_nm`, spectral full width at half maximum,
- `tau_moment_ns_raw`, moment lifetime,
- `tau_ampw_ns_raw`, amplitude weighted lifetime,
- `tau_intw_ns_raw`, intensity weighted lifetime.

## Aggregation

Repeated spectra for the same quantum dot and temperature are aggregated with the median. This reduces repeated-measure bias and produces one row per `(qd_id, temperature_K)`.

## Validation

Train and test sets are split with `GroupShuffleSplit` using `qd_id` as the group. This prevents the same quantum dot from appearing in both training and test sets.

## Models

The workflow compares a constant baseline, a Ridge regression model and a Random Forest. Optional hyperparameter tuning is performed with Dask-backed `GridSearchCV`.
