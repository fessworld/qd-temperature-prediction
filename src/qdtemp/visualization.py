from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def plot_predicted_vs_true(y_true, y_pred, title: str, output_path: str | Path) -> None:
    """Save a predicted vs true temperature scatter plot."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(5, 5))
    plt.scatter(y_true, y_pred, s=18, alpha=0.6)
    lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    plt.plot(lims, lims, "k-", linewidth=1)
    plt.xlabel("True temperature (K)")
    plt.ylabel("Predicted temperature (K)")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_feature_importances(importances: pd.Series, title: str, output_path: str | Path) -> None:
    """Save a feature importance bar plot."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(6, 4))
    importances.sort_values(ascending=False).plot(kind="bar")
    plt.title(title)
    plt.ylabel("Importance")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
