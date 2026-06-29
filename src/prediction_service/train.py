from __future__ import annotations

import copy
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim
from sklearn.metrics import f1_score, mean_squared_error, roc_auc_score
from torch.optim.lr_scheduler import ReduceLROnPlateau

from . import model as model_module
from . import sequences as seq_module
from .paths import FIGURES_DIR, MODELS_DIR, ONNX_DIR

try:
    import mlflow

    MLFLOW_AVAILABLE = True
except Exception:
    mlflow = None
    MLFLOW_AVAILABLE = False


def configure_mlflow() -> bool:
    if not MLFLOW_AVAILABLE:
        return False

    try:
        mlflow.set_tracking_uri("http://mlflow-service.argocd.svc.cluster.local:5000")
        mlflow.set_experiment("aerospace-predictive-maintenance")
        mlflow.pytorch.autolog()
        return True
    except Exception as exc:
        print(f"MLflow disabled: {exc}")
        return False


def save_learning_curves(history: dict[str, list[float]]) -> Path:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    output_path = FIGURES_DIR / "learning_curves.png"

    plt.figure(figsize=(10, 5))
    plt.plot(history["train_loss"], label="Train Loss")
    plt.plot(history["val_loss"], label="Val Loss")
    plt.title("Training and Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss (alpha*MSE + beta*BCE)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    return output_path


def export_onnx(model: torch.nn.Module, device: torch.device) -> Path:
    ONNX_DIR.mkdir(parents=True, exist_ok=True)
    output_path = ONNX_DIR / "multitask_model.onnx"
    dummy_input = torch.randn(1, 30, 14, device=device)

    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["rul_output", "alert_output"],
        dynamic_axes={
            "input": {0: "batch_size"},
            "rul_output": {0: "batch_size"},
            "alert_output": {0: "batch_size"},
        },
    )
    return output_path


def main() -> None:
    use_mlflow = configure_mlflow()

    train_loader = seq_module.train_loader
    val_loader = seq_module.val_loader

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = model_module.MultiTaskModel(n_features=14, window_size=30).to(device)

    epochs = 100
    lr = 1e-3
    weight_decay = 1e-4
    alpha = 1.0
    beta = 100.0
    es_patience = 10

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)

    best_val_loss = float("inf")
    best_model_weights = copy.deepcopy(model.state_dict())
    es_counter = 0

    history = {"train_loss": [], "val_loss": [], "val_rmse": [], "val_f1": [], "val_auc": []}

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    if use_mlflow:
        mlflow.start_run()

    print("\n--- Starting Training Loop ---")

    for epoch in range(epochs):
        model.train()
        train_loss_epoch = 0.0

        for X_batch, rul_batch, alert_batch in train_loader:
            X_batch = X_batch.to(device)
            rul_batch = rul_batch.to(device)
            alert_batch = alert_batch.to(device)

            optimizer.zero_grad()
            rul_pred, alert_pred = model(X_batch)
            total_loss, _, _ = model_module.combined_loss(
                rul_pred, rul_batch, alert_pred, alert_batch, alpha=alpha, beta=beta
            )
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss_epoch += total_loss.item() * X_batch.size(0)

        train_loss_epoch /= len(train_loader.dataset)

        model.eval()
        val_loss_epoch = 0.0
        all_rul_preds, all_rul_trues = [], []
        all_alert_preds, all_alert_trues = [], []

        with torch.no_grad():
            for X_batch, rul_batch, alert_batch in val_loader:
                X_batch = X_batch.to(device)
                rul_batch = rul_batch.to(device)
                alert_batch = alert_batch.to(device)

                rul_pred, alert_pred = model(X_batch)
                total_loss, _, _ = model_module.combined_loss(
                    rul_pred, rul_batch, alert_pred, alert_batch, alpha=alpha, beta=beta
                )

                val_loss_epoch += total_loss.item() * X_batch.size(0)
                all_rul_preds.extend(rul_pred.cpu().numpy())
                all_rul_trues.extend(rul_batch.cpu().numpy())
                all_alert_preds.extend(alert_pred.cpu().numpy())
                all_alert_trues.extend(alert_batch.cpu().numpy())

        val_loss_epoch /= len(val_loader.dataset)
        rmse = np.sqrt(mean_squared_error(all_rul_trues, all_rul_preds))
        alert_preds_binary = (np.array(all_alert_preds) >= 0.5).astype(int)

        try:
            auc = roc_auc_score(all_alert_trues, all_alert_preds)
        except ValueError:
            auc = 0.5

        f1 = f1_score(all_alert_trues, alert_preds_binary, zero_division=0)

        history["train_loss"].append(train_loss_epoch)
        history["val_loss"].append(val_loss_epoch)
        history["val_rmse"].append(rmse)
        history["val_f1"].append(f1)
        history["val_auc"].append(auc)

        if use_mlflow:
            mlflow.log_metric("train_loss", train_loss_epoch, step=epoch)
            mlflow.log_metric("val_loss", val_loss_epoch, step=epoch)
            mlflow.log_metric("val_rmse", rmse, step=epoch)
            mlflow.log_metric("val_f1", f1, step=epoch)
            mlflow.log_metric("val_auc", auc, step=epoch)

        print(
            f"Epoch {epoch + 1:03d}/{epochs} | "
            f"Train Loss: {train_loss_epoch:.2f} | "
            f"Val Loss: {val_loss_epoch:.2f} | "
            f"Val RMSE: {rmse:.2f} | Val F1: {f1:.3f} | Val AUC: {auc:.3f}"
        )

        scheduler.step(val_loss_epoch)

        if val_loss_epoch < best_val_loss:
            best_val_loss = val_loss_epoch
            best_model_weights = copy.deepcopy(model.state_dict())
            es_counter = 0
            torch.save(best_model_weights, MODELS_DIR / "best_multitask_model.pth")
        else:
            es_counter += 1
            if es_counter >= es_patience:
                print(f"\nEarly stopping triggered at epoch {epoch + 1}.")
                break

    model.load_state_dict(best_model_weights)

    curves_path = save_learning_curves(history)
    onnx_path = export_onnx(model.cpu(), torch.device("cpu"))

    if use_mlflow:
        mlflow.log_artifact(str(curves_path))
        mlflow.log_artifact(str(MODELS_DIR / "best_multitask_model.pth"), artifact_path="model")
        mlflow.log_artifact(str(onnx_path))
        mlflow.end_run()

    print(f"Saved best weights to {MODELS_DIR / 'best_multitask_model.pth'}")
    print(f"Saved learning curves to {curves_path}")
    print(f"Exported ONNX model to {onnx_path}")


if __name__ == "__main__":
    main()
