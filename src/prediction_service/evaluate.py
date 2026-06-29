from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import auc, f1_score, mean_squared_error, precision_recall_curve, roc_auc_score, roc_curve

from . import model as model_module
from .paths import FIGURES_DIR, MODELS_DIR, DATA_PROCESSED_DIR

try:
    import shap  # noqa: F401
except Exception:
    shap = None


def get_nasa_score(y_true, y_pred):
    diff = y_pred - y_true
    score = 0.0
    for d in diff:
        if d < 0:
            score += np.exp(-d / 13.0) - 1
        else:
            score += np.exp(d / 10.0) - 1
    return score


def get_dataset_windows(dataset_name: str):
    test_df = pd.read_csv(DATA_PROCESSED_DIR / f"test_{dataset_name}_labeled.csv")
    feature_cols = [c for c in test_df.columns if c not in ["unit", "cycle", "RUL", "alert"]]
    return test_df, feature_cols


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device for Evaluation: {device}")

    model = model_module.MultiTaskModel(n_features=14, window_size=30).to(device)
    model.load_state_dict(torch.load(MODELS_DIR / "best_multitask_model.pth", map_location=device))
    model.eval()

    print("\n--- Evaluate Generalization Across Datasets ---")

    feature_names = None
    datasets = ["FD001", "FD002", "FD003", "FD004"]
    results = []
    fd001_payload = None

    for fd in datasets:
        test_df, feature_cols = get_dataset_windows(fd)
        if feature_names is None:
            feature_names = feature_cols

        final_windows, final_ruls, final_alerts = [], [], []
        engine_ids = []

        for unit_id, group in test_df.groupby("unit"):
            data = group[feature_cols].values
            rul = group["RUL"].values
            alert = group["alert"].values

            if len(data) >= 30:
                final_windows.append(data[-30:, :])
                final_ruls.append(rul[-1])
                final_alerts.append(alert[-1])
                engine_ids.append(unit_id)

        if not final_windows:
            continue

        x_test = torch.tensor(np.array(final_windows), dtype=torch.float32).to(device)

        with torch.no_grad():
            rul_pred, alert_pred = model(x_test)

        rul_pred_np = rul_pred.cpu().numpy()
        alert_pred_np = alert_pred.cpu().numpy()

        rmse = np.sqrt(mean_squared_error(final_ruls, rul_pred_np))
        nasa_score = get_nasa_score(np.array(final_ruls), rul_pred_np)

        alert_pred_bin = (alert_pred_np >= 0.5).astype(int)
        f1 = f1_score(final_alerts, alert_pred_bin, zero_division=0)
        try:
            roc_auc = roc_auc_score(final_alerts, alert_pred_np)
            precision, recall, _ = precision_recall_curve(final_alerts, alert_pred_np)
            pr_auc = auc(recall, precision)
        except ValueError:
            roc_auc, pr_auc = 0.5, 0.5

        results.append(
            {
                "Dataset": fd,
                "RMSE": rmse,
                "NASA_Score": nasa_score,
                "F1_Score": f1,
                "ROC_AUC": roc_auc,
                "PR_AUC": pr_auc,
            }
        )

        if fd == "FD001":
            fd001_payload = {
                "final_alerts": final_alerts,
                "alert_pred_np": alert_pred_np,
                "test_df": test_df,
                "feature_cols": feature_cols,
            }

    res_df = pd.DataFrame(results)
    print(res_df.to_string(index=False, float_format="%.2f"))
    print()

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    if fd001_payload is not None:
        print("--- Plotting Precision-Recall Curve & ROC Curve for FD001 ---")
        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        precision, recall, _ = precision_recall_curve(
            fd001_payload["final_alerts"], fd001_payload["alert_pred_np"]
        )
        plt.plot(recall, precision, marker=".")
        plt.title("Precision-Recall Curve (Alert Head - FD001)")
        plt.xlabel("Recall")
        plt.ylabel("Precision")

        plt.subplot(1, 2, 2)
        fpr, tpr, _ = roc_curve(fd001_payload["final_alerts"], fd001_payload["alert_pred_np"])
        plt.plot(fpr, tpr, marker=".")
        plt.title("ROC Curve (Alert Head - FD001)")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")

        plt.tight_layout()
        plt.savefig(FIGURES_DIR / "classification_eval_fd001.png")
        plt.close()

        print("--- Plot predicted vs true RUL per engine ---")
        longest_engine = fd001_payload["test_df"]["unit"].value_counts().idxmax()
        engine_plot = fd001_payload["test_df"][fd001_payload["test_df"]["unit"] == longest_engine].copy()
        engine_data = engine_plot[fd001_payload["feature_cols"]].values
        engine_ruls = engine_plot["RUL"].values

        win_preds = []
        win_trues = []
        win_cycles = engine_plot["cycle"].values[29:]

        for i in range(len(engine_data) - 30 + 1):
            win = torch.tensor(engine_data[np.newaxis, i : i + 30, :], dtype=torch.float32).to(device)
            with torch.no_grad():
                r, _ = model(win)
            win_preds.append(r.item())
            win_trues.append(engine_ruls[i + 29])

        plt.figure(figsize=(10, 5))
        plt.plot(win_cycles, win_trues, label="True RUL", color="blue")
        plt.plot(win_cycles, win_preds, label="Predicted RUL", color="red", linestyle="--")
        plt.xlabel("Cycle")
        plt.ylabel("Remaining Useful Life (RUL)")
        plt.title(f"Predicted vs True RUL (FD001 - Test Engine {longest_engine})")
        plt.legend()
        plt.grid()
        plt.savefig(FIGURES_DIR / "rul_prediction_engine1.png")
        plt.close()

        print("--- Use feature permutation to estimate importance ---")
        try:
            baseline_rmse = rmse
            feature_importances = []
            x_test_tz = x_test

            for j in range(len(feature_names)):
                test_pert = x_test_tz.clone()
                idx = torch.randperm(test_pert.shape[0])
                test_pert[:, :, j] = test_pert[idx, :, j]

                with torch.no_grad():
                    r_pred, _ = model(test_pert)

                r_pred_np = r_pred.cpu().numpy()
                p_rmse = np.sqrt(mean_squared_error(fd001_payload["final_alerts"], r_pred_np))
                feature_importances.append(p_rmse - baseline_rmse)

            mean_shap = np.clip(np.array(feature_importances), a_min=0, a_max=None)
            indices = np.argsort(mean_shap)

            plt.figure(figsize=(10, 8))
            plt.barh(range(len(indices)), mean_shap[indices], align="center")
            plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
            plt.xlabel("Mean |SHAP| Value (Impact on RUL output)")
            plt.title("Feature Importance (Sensor driving RUL predictions)")
            plt.tight_layout()
            plt.savefig(FIGURES_DIR / "shap_feature_importance.png")
            plt.close()
        except Exception as exc:
            print(f"Feature importance step skipped: {exc}")

    print("Step 7 Complete!")


if __name__ == "__main__":
    main()
