# Data folder

Do not commit raw experimental files to this public repository unless you have permission to share them.

For public demos, generate synthetic data with:

```bash
python scripts/make_synthetic_demo_data.py --out data/features_with_tau_real.csv
```

For real use, place a CSV file with at least these columns:

```text
qd_id, temperature_K, peak_eV, fwhm_nm,
tau_moment_ns_raw, tau_ampw_ns_raw, tau_intw_ns_raw
```

The training script groups rows by `(qd_id, temperature_K)` and uses median feature values for each dot and temperature.
