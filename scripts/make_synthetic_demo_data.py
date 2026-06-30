from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def make_demo_data(n_qds: int = 120, spectra_per_qd: int = 4, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    temps = np.array([10, 20, 40, 60, 80, 100, 110], dtype=float)
    rows = []
    for qd_id in range(1, n_qds + 1):
        temp = float(rng.choice(temps))
        qd_shift = rng.normal(0, 0.012)
        for time_idx in range(spectra_per_qd):
            peak_eV = 2.42 - 0.00035 * temp + qd_shift + rng.normal(0, 0.004)
            fwhm_nm = 0.30 + 0.0065 * temp + rng.normal(0, 0.08)
            tau_moment = 5.0 + 0.020 * temp + rng.normal(0, 0.8)
            tau_ampw = 0.8 + 0.030 * temp + rng.normal(0, 0.7)
            tau_intw = 14.0 + 0.015 * temp + rng.normal(0, 1.2)
            rows.append({
                "qd_id": qd_id,
                "temperature_K": temp,
                "peak_eV": peak_eV,
                "fwhm_nm": max(fwhm_nm, 0.05),
                "tau_moment_ns_raw": max(tau_moment, 0.05),
                "tau_ampw_ns_raw": max(tau_ampw, 0.05),
                "tau_intw_ns_raw": max(tau_intw, 0.05),
            })
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic quantum dot feature data.")
    parser.add_argument("--out", type=Path, default=Path("data/features_with_tau_real.csv"))
    parser.add_argument("--n-qds", type=int, default=120)
    parser.add_argument("--spectra-per-qd", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    df = make_demo_data(args.n_qds, args.spectra_per_qd, args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"Saved {len(df)} rows to {args.out}")


if __name__ == "__main__":
    main()
