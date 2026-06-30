from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd
from dask.distributed import Client, LocalCluster

from qdtemp.model import (
    evaluate_predictions,
    group_split,
    load_feature_table,
    make_models,
    prepare_modeling_table,
    run_dask_grid_search,
)
from qdtemp.visualization import plot_feature_importances, plot_predicted_vs_true


def main() -> None:
    parser = argparse.ArgumentParser(description="Train temperature prediction models.")
    parser.add_argument("--input", type=Path, default=Path("data/features_with_tau_real.csv"))
    parser.add_argument("--output", type=Path, default=Path("outputs/temp_prediction"))
    parser.add_argument("--use-dask-grid", action="store_true")
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)

    df = load_feature_table(args.input)
    d_model, feature_cols = prepare_modeling_table(df)
    print("Modeling table shape:", d_model.shape)
    print("Features:", feature_cols)

    split = group_split(d_model, feature_cols, random_state=args.random_state)
    models = make_models(feature_cols, random_state=args.random_state)

    metrics = []
    for name, model in models.items():
        print(f"Training {name}")
        t0 = time.perf_counter()
        model.fit(split.X_train, split.y_train)
        train_time = time.perf_counter() - t0

        t0 = time.perf_counter()
        y_pred = model.predict(split.X_test)
        predict_time = time.perf_counter() - t0

        row = {
            "model": name,
            **evaluate_predictions(split.y_test, y_pred),
            "train_time_s": train_time,
            "predict_time_s": predict_time,
            "n_train": len(split.X_train),
            "n_test": len(split.X_test),
        }
        metrics.append(row)
        plot_predicted_vs_true(
            split.y_test,
            y_pred,
            f"{name} predicted vs true",
            args.output / f"pred_vs_true_{name}.png",
        )

    metrics_df = pd.DataFrame(metrics).sort_values("RMSE_K")
    metrics_df.to_csv(args.output / "metrics_baseline.csv", index=False)
    print(metrics_df)

    rf_model = models["RandomForest"]
    importances = pd.Series(rf_model.feature_importances_, index=feature_cols).sort_values(ascending=False)
    importances.to_csv(args.output / "feature_importances_random_forest.csv", header=["importance"])
    plot_feature_importances(
        importances,
        "RandomForest feature importances",
        args.output / "feature_importances_random_forest.png",
    )

    if args.use_dask_grid:
        cluster = LocalCluster(n_workers=4, threads_per_worker=1, memory_limit="2GB")
        client = Client(cluster)
        print(client)
        try:
            t0 = time.perf_counter()
            grid = run_dask_grid_search(split, random_state=args.random_state)
            grid_time = time.perf_counter() - t0
            best = grid.best_estimator_
            y_pred = best.predict(split.X_test)
            grid_row = {
                "model": "RandomForest_GridSearch_Dask",
                **evaluate_predictions(split.y_test, y_pred),
                "grid_search_time_s": grid_time,
                "n_train": len(split.X_train),
                "n_test": len(split.X_test),
                "best_params": str(grid.best_params_),
            }
            pd.DataFrame([grid_row]).to_csv(args.output / "metrics_rf_grid_dask.csv", index=False)
            plot_predicted_vs_true(
                split.y_test,
                y_pred,
                "RandomForest Dask GridSearch predicted vs true",
                args.output / "pred_vs_true_RandomForest_grid_dask.png",
            )
            importances_best = pd.Series(best.feature_importances_, index=feature_cols).sort_values(ascending=False)
            importances_best.to_csv(args.output / "feature_importances_random_forest_grid_dask.csv", header=["importance"])
            plot_feature_importances(
                importances_best,
                "RandomForest Dask GridSearch feature importances",
                args.output / "feature_importances_random_forest_grid_dask.png",
            )
        finally:
            client.close()
            cluster.close()


if __name__ == "__main__":
    main()
