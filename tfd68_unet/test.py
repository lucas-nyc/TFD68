import os
import argparse
import json
from typing import Optional, List
import glob
import numpy as np
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import cv2
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

import config.config as config

from dataset.datasetloader import build_xy_dataset

from utils.utils import (
        landmarks_to_pixels,
        build_metrics_tables,
        compute_nme,
        plot_landmark_samples,
    )

from utils.loss import NormalizedMeanError, wing_loss

def choose_random_indices(N: int, num_samples: int, seed: Optional[int] = None) -> List[int]:
    if N <= 0:
        return []
    n = min(num_samples, N)
    if n == N:
        return list(range(N))
    rng = np.random.default_rng(seed)
    return rng.choice(np.arange(N), size=n, replace=False).tolist()

def load_model_with_customs(model_path: str):
    custom_objects = {'wing_loss': wing_loss}
    if NormalizedMeanError is not None:
        custom_objects['NormalizedMeanError'] = NormalizedMeanError
    return tf.keras.models.load_model(model_path, custom_objects=custom_objects, compile=False)

def _ensure_out_run_dir(base_out_dir: str, run_prefix: str = "run"):
    os.makedirs(base_out_dir, exist_ok=True)
    existing = [d for d in os.listdir(base_out_dir) if os.path.isdir(os.path.join(base_out_dir, d)) and d.startswith(run_prefix)]
    idxs = []
    for e in existing:
        try:
            idxs.append(int(e.replace(run_prefix, "")))
        except Exception:
            pass
    next_idx = max(idxs) + 1 if idxs else 1
    out_run = os.path.join(base_out_dir, f"{run_prefix}{next_idx}")
    os.makedirs(out_run, exist_ok=True)
    return out_run

def find_stage_model(stage: str, search_root: str) -> Optional[str]:
    patterns = [
        os.path.join(search_root, f"**/*{stage}*best*.h5"),
        os.path.join(search_root, f"**/*{stage}*.h5"),
    ]
    candidates = []
    for pat in patterns:
        candidates += glob.glob(pat, recursive=True)
    if not candidates:
        return None
    candidates = sorted(candidates, key=lambda p: os.path.getmtime(p), reverse=True)
    return candidates[0]

def preds_to_pixel_coords(flat_preds: np.ndarray, H: int, W: int):
    arr = np.asarray(flat_preds).reshape(-1, 2)
    xs = arr[:, 0] * W
    ys = arr[:, 1] * H
    return xs, ys

def compute_test_nme_batches(model: tf.keras.Model,
                             lm_ds_test: tf.data.Dataset,
                             num_batches: int = 50):
    try:
        collected = collect_predictions_from_dataset_streaming(model, lm_ds_test, max_batches=num_batches)
        if collected is None:
            print(f"[compute_test_nme_batches] no data collected for {num_batches} batches")
            return None
        X_all, y_lm_all, y_expr_all, pred_lm_all, pred_expr_all = collected

        per_sample_nme = safe_compute_nme(pred_lm_all, y_lm_all, X_images=X_all)
        mean_nme = float(np.nanmean(per_sample_nme)) if per_sample_nme.size > 0 else float("nan")
        return mean_nme, per_sample_nme
    except Exception as e:
        print(f"[compute_test_nme_batches] failed: {e}")
        return None

def collect_predictions_from_dataset_streaming(model: tf.keras.Model,
                                               lm_ds_test: tf.data.Dataset,
                                               max_batches: Optional[int] = None):
    X_list, y_lm_list, y_expr_list = [], [], []
    pred_lm_list, pred_expr_list = [], []
    for i, batch in enumerate(lm_ds_test):
        if max_batches is not None and i >= max_batches:
            break
        if isinstance(batch, tuple) and len(batch) == 3:
            x_batch, y_batch, _ = batch
        else:
            x_batch, y_batch = batch

        x_np = x_batch.numpy() if isinstance(x_batch, tf.Tensor) else np.asarray(x_batch)

        if isinstance(y_batch, dict):
            y_lm_np = y_batch.get("landmark_output")
            y_expr_np = y_batch.get("expression_output")
            y_lm_np = y_lm_np.numpy() if isinstance(y_lm_np, tf.Tensor) else np.asarray(y_lm_np)
            y_expr_np = y_expr_np.numpy() if isinstance(y_expr_np, tf.Tensor) else np.asarray(y_expr_np)
        else:
            try:
                y_lm = y_batch[0]
                y_expr = y_batch[1]
            except Exception:
                raise RuntimeError("Unexpected y_batch format: expected dict or tuple (landmark, expression)")
            y_lm_np = y_lm.numpy() if isinstance(y_lm, tf.Tensor) else np.asarray(y_lm)
            y_expr_np = y_expr.numpy() if isinstance(y_expr, tf.Tensor) else np.asarray(y_expr)

        preds = model.predict_on_batch(x_batch)
        if isinstance(preds, (list, tuple)):
            pred_lm_np, pred_expr_np = preds[0], preds[1]
        elif isinstance(preds, dict):
            pred_lm_np = preds.get('landmark_output')
            pred_expr_np = preds.get('expression_output')
        else:
            raise RuntimeError("Unexpected model.predict output for two-head model.")

        pred_lm_np = np.asarray(pred_lm_np)
        pred_expr_np = np.asarray(pred_expr_np)

        X_list.append(x_np)
        y_lm_list.append(y_lm_np)
        y_expr_list.append(y_expr_np)
        pred_lm_list.append(pred_lm_np)
        pred_expr_list.append(pred_expr_np)

    if len(X_list) == 0:
        return None

    X_all = np.concatenate(X_list, axis=0)
    y_lm_all = np.concatenate(y_lm_list, axis=0)
    y_expr_all = np.concatenate(y_expr_list, axis=0)
    pred_lm_all = np.concatenate(pred_lm_list, axis=0)
    pred_expr_all = np.concatenate(pred_expr_list, axis=0)
    return X_all, y_lm_all, y_expr_all, pred_lm_all, pred_expr_all

def evaluate_mask_model_stream(mask_model: tf.keras.Model,
                               mask_ds_test: tf.data.Dataset,
                               out_dir: str,
                               prefix: str = "mask_test",
                               batch_size: int = 32,
                               thr: float = 0.5,
                               max_visualize: int = 20,
                               max_batches: Optional[int] = None):
    os.makedirs(out_dir, exist_ok=True)
    viz_dir = os.path.join(out_dir, f"{prefix}_viz")
    os.makedirs(viz_dir, exist_ok=True)
    ious = []

    sample_idx = 0
    saved = 0
    for i, batch in enumerate(mask_ds_test):
        if max_batches is not None and i >= max_batches:
            break
        if isinstance(batch, tuple) and len(batch) == 3:
            x_batch, y_batch, _ = batch
        else:
            x_batch, y_batch = batch
        if isinstance(x_batch, dict):
            x_in = x_batch.get("image", list(x_batch.values())[0])
        else:
            x_in = x_batch
        if isinstance(y_batch, dict):
            mask_true = y_batch.get("mask", None) or y_batch.get("mask_output", None)
            if mask_true is None:
                raise RuntimeError("Mask dataset y dict does not contain 'mask' or 'mask_output'.")
        else:
            mask_true = y_batch

        preds = mask_model.predict_on_batch(x_in)
        preds = np.asarray(preds)
        mask_true_np = np.asarray(mask_true)
        if preds.ndim == 3:
            preds = np.expand_dims(preds, -1)
        if mask_true_np.ndim == 3:
            mask_true_np = np.expand_dims(mask_true_np, -1)

        batch_ious = []
        for p, t in zip(preds, mask_true_np):
            pbin = (p.squeeze() >= thr).astype(np.uint8)
            tbin = (t.squeeze() >= 0.5).astype(np.uint8)
            inter = np.logical_and(pbin, tbin).sum()
            union = np.logical_or(pbin, tbin).sum()
            batch_ious.append(float(inter) / (float(union) + 1e-12))
        ious.extend(batch_ious)

        x_np = x_in.numpy() if isinstance(x_in, tf.Tensor) else np.asarray(x_in)
        B = preds.shape[0]
        for bi in range(B):
            if saved >= max_visualize:
                break
            crop = x_np[bi]
            if crop.ndim == 3 and crop.shape[2] >= 1:
                im_gray = crop[:, :, 0]
            else:
                im_gray = crop
            im_vis = (im_gray * 255.0).astype(np.uint8) if im_gray.max() <= 1.01 else im_gray.astype(np.uint8)
            im_vis = cv2.cvtColor(im_vis, cv2.COLOR_GRAY2BGR)
            pmask = (preds[bi].squeeze() >= thr).astype(np.uint8) * 255
            tmask = (mask_true_np[bi].squeeze() >= 0.5).astype(np.uint8) * 255
            overlay = im_vis.copy()
            overlay[pmask > 0] = (0, 255, 0)
            overlay[tmask > 0] = (255, 0, 0)
            out_p = os.path.join(viz_dir, f"{prefix}_{sample_idx:06d}_mask_overlay.png")
            cv2.imwrite(out_p, overlay)
            saved += 1
            sample_idx += 1
    if len(ious) == 0:
        print("[mask eval] no samples")
        return None
    ious = np.array(ious, dtype=float)
    pd.DataFrame({"iou": ious}).to_csv(os.path.join(out_dir, f"{prefix}_ious.csv"), index=False)
    print(f"[mask eval] mean IoU: {ious.mean():.4f} (N={len(ious)})")
    print("Mask visualizations ->", viz_dir)
    return ious

def evaluate_landmark_stage(lmk_model: tf.keras.Model,
                            lm_ds_test: tf.data.Dataset,
                            out_dir: str,
                            prefix: str = "landmark_test",
                            max_visualize: int = 30,
                            max_batches: Optional[int] = None):
    os.makedirs(out_dir, exist_ok=True)
    viz_dir = os.path.join(out_dir, f"{prefix}_viz")
    os.makedirs(viz_dir, exist_ok=True)

    class LMWrapper:
        def __init__(self, model, num_classes):
            self.model = model
            self.num_classes = num_classes
        def predict_on_batch(self, x):
            lm = self.model.predict_on_batch(x)
            dummy = np.zeros((lm.shape[0], self.num_classes), dtype=float)
            return [lm, dummy]

    wrapper = LMWrapper(lmk_model, getattr(config, "NUM_CLASSES", 8))

    collected = collect_predictions_from_dataset_streaming(wrapper, lm_ds_test, max_batches=max_batches)
    if collected is None:
        print("[landmark eval] no data")
        return None
    X_all, y_lm_all, y_expr_all, pred_lm_all, pred_expr_all = collected

    per_sample_nme = safe_compute_nme(pred_lm_all, y_lm_all, X_images=X_all)

    mean_nme = float(np.nanmean(per_sample_nme))
    print(f"[landmark eval] mean NME: {mean_nme:.6f} (N={len(per_sample_nme)})")

    num_to_plot = min(max_visualize, X_all.shape[0])
    rand_indices = choose_random_indices(X_all.shape[0], num_to_plot, seed=None)
    saved = plot_landmark_samples(X_all, pred_lm_all, pred_expr_all, y_true_landmarks=y_lm_all, y_true_expr=y_expr_all, nme=per_sample_nme,
                                      indices=rand_indices,
                                      num_samples=num_to_plot, out_dir=viz_dir, show=False)
    print(f"[landmark eval] Saved {len(saved)} sample visualizations -> {viz_dir}")

    pd.DataFrame({"nme": per_sample_nme}).to_csv(os.path.join(out_dir, f"{prefix}_nme.csv"), index=False)
    return {"mean_nme": mean_nme, "per_sample_nme": per_sample_nme}

def safe_compute_nme(pred_lm_all, y_lm_all, X_images=None, normalize_by='interocular'):
    if compute_nme is not None:
        try:
            nme_res = compute_nme(pred_lm_all, y_lm_all, X_images=X_images, normalize_by=normalize_by, print_summary=False)
            per_sample_nme = nme_res['per_sample_nme']
            return per_sample_nme
        except Exception:
            pass

    if X_images is None:
        raise RuntimeError("safe_compute_nme fallback requires X_images when compute_nme is unavailable.")
    H, W = X_images.shape[1], X_images.shape[2]
    if landmarks_to_pixels is not None:
        true_px = landmarks_to_pixels(y_lm_all, H=H, W=W)
        pred_px = landmarks_to_pixels(pred_lm_all, H=H, W=W)
    else:
        true_px = np.asarray(y_lm_all).reshape(y_lm_all.shape[0], -1, 2)
        pred_px = np.asarray(pred_lm_all).reshape(pred_lm_all.shape[0], -1, 2)
        true_px[..., 0] *= W; true_px[..., 1] *= H
        pred_px[..., 0] *= W; pred_px[..., 1] *= H

    dists = np.linalg.norm(pred_px - true_px, axis=2)
    mean_pix = dists.mean(axis=1)

    iods = np.zeros((X_images.shape[0],), dtype=float)
    for i in range(X_images.shape[0]):
        t = true_px[i]
        xmin, ymin = t[:,0].min(), t[:,1].min()
        xmax, ymax = t[:,0].max(), t[:,1].max()
        iod = np.sqrt((xmax-xmin)**2 + (ymax-ymin)**2)
        iods[i] = iod if iod > 1e-6 else 1.0
    per_sample_nme = mean_pix / (iods + 1e-12)
    return per_sample_nme

def evaluate_final_stage(final_model: tf.keras.Model,
                         lm_ds_test: tf.data.Dataset,
                         out_dir: str,
                         entries: Optional[List[dict]] = None,
                         prefix: str = "final_test",
                         max_visualize: int = 100,
                         max_batches: Optional[int] = None):
    os.makedirs(out_dir, exist_ok=True)
    viz_dir = os.path.join(out_dir, f"{prefix}_viz")
    os.makedirs(viz_dir, exist_ok=True)

    num_classes = int(getattr(config, "NUM_CLASSES", 8))

    collected = collect_predictions_from_dataset_streaming(final_model, lm_ds_test, max_batches=max_batches)
    if collected is None:
        print("[final eval] no data")
        return None
    X_all, y_lm_all, y_expr_all, pred_lm_all, pred_expr_all = collected
    N = X_all.shape[0]
    print(f"[final eval] Collected {N} samples.")

    per_sample_nme = safe_compute_nme(pred_lm_all, y_lm_all, X_images=X_all)
    mean_nme = float(np.nanmean(per_sample_nme))
    median_nme = float(np.nanmedian(per_sample_nme))
    print(f"[final eval] mean NME: {mean_nme:.6f}, median NME: {median_nme:.6f}")

    if pred_expr_all.ndim == 2 and pred_expr_all.shape[1] > 1:
        pred_internal = np.argmax(pred_expr_all, axis=1)
        pred_P = pred_expr_all.shape[1]
    else:
        pred_internal = pred_expr_all.reshape(-1).astype(int)
        pred_P = 1
    if y_expr_all.ndim == 2 and y_expr_all.shape[1] > 1:
        true_labels_full = np.argmax(y_expr_all, axis=1)
    else:
        true_labels_full = y_expr_all.reshape(-1).astype(int)
    if pred_P == num_classes - 1:
        pred_display = pred_internal + 1
        zeros_col = np.zeros((pred_expr_all.shape[0], 1), dtype=pred_expr_all.dtype)
        pred_expr_for_metrics = np.concatenate([zeros_col, pred_expr_all], axis=1)
    else:
        pred_display = pred_internal
        pred_expr_for_metrics = pred_expr_all

    if pred_expr_for_metrics is None:
        pred_expr_for_metrics = np.zeros((pred_display.shape[0], num_classes), dtype=float)
    if pred_expr_for_metrics.ndim == 1:
        onehot = np.zeros((pred_expr_for_metrics.shape[0], num_classes), dtype=float)
        onehot[np.arange(onehot.shape[0]), pred_expr_for_metrics] = 1.0
        pred_expr_for_metrics = onehot
    elif pred_expr_for_metrics.ndim == 2 and pred_expr_for_metrics.shape[1] == num_classes - 1:
        pred_expr_for_metrics = np.concatenate([np.zeros((pred_expr_for_metrics.shape[0],1)), pred_expr_for_metrics], axis=1)

    if pred_display.shape[0] != true_labels_full.shape[0]:
        raise RuntimeError(f"Mismatch in predicted vs true expr counts: {pred_display.shape[0]} vs {true_labels_full.shape[0]}")

    labels_range = list(np.arange(num_classes))
    cm_full = confusion_matrix(true_labels_full, pred_display, labels=labels_range)
    report_full = classification_report(true_labels_full, pred_display, digits=4, zero_division=0)
    acc_full = accuracy_score(true_labels_full, pred_display)
    print(f"[final eval] Expr accuracy (full incl class0): {acc_full:.4f}")
    print(report_full)
    pd.DataFrame(cm_full).to_csv(os.path.join(out_dir, f"{prefix}_expr_confusion_full.csv"), index=False)
    with open(os.path.join(out_dir, f"{prefix}_expr_report_full.txt"), "w") as f:
        f.write(report_full); f.write(f"\nAccuracy: {acc_full:.6f}\n")

    mask_non0 = (true_labels_full != 0)
    if np.any(mask_non0):
        true_labels_filtered = true_labels_full[mask_non0]
        pred_labels_filtered = pred_display[mask_non0]
        labels_filt = list(np.arange(1, num_classes))
        cm_filt = confusion_matrix(true_labels_filtered, pred_labels_filtered, labels=labels_filt)
        report_filt = classification_report(true_labels_filtered, pred_labels_filtered, digits=4, zero_division=0)
        acc_filt = accuracy_score(true_labels_filtered, pred_labels_filtered)
        mask_non0 = (true_labels_full != 0)
        if np.any(mask_non0):
            true_labels_filtered = true_labels_full[mask_non0]
            pred_labels_filtered = pred_display[mask_non0]
            labels_filt = list(np.arange(1, num_classes))
            cm_filt = confusion_matrix(true_labels_filtered, pred_labels_filtered, labels=labels_filt)
            report_filt = classification_report(true_labels_filtered, pred_labels_filtered, digits=4, zero_division=0)
            acc_filt = accuracy_score(true_labels_filtered, pred_labels_filtered)
            print(f"[final eval] Expr accuracy (filtered exclude class0): {acc_filt:.4f}")
            print(report_filt)
            pd.DataFrame(cm_filt).to_csv(os.path.join(out_dir, f"{prefix}_expr_confusion_filtered.csv"), index=False)
            with open(os.path.join(out_dir, f"{prefix}_expr_report_filtered.txt"), "w") as f:
                f.write(report_filt); f.write(f"\nAccuracy: {acc_filt:.6f}\n")

            plt.figure(figsize=(6, 5))
            plt.imshow(cm_filt, interpolation='nearest', cmap=plt.cm.Blues)
            plt.title(f"Confusion Matrix")
            plt.colorbar()
            plt.xlabel("Predicted")
            plt.ylabel("True")
            labels_plot = [str(i) for i in labels_filt]
            tick_marks = np.arange(len(labels_plot))
            plt.xticks(tick_marks, labels_plot, rotation=45)
            plt.yticks(tick_marks, labels_plot)
            for i in range(cm_filt.shape[0]):
                for j in range(cm_filt.shape[1]):
                    plt.text(j, i, int(cm_filt[i, j]), ha="center", va="center",
                            color="white" if cm_filt[i, j] > cm_filt.max() / 2.0 else "black")
            plt.tight_layout()
            cm_filt_png = os.path.join(out_dir, f"{prefix}_confusion_filtered.png")
            plt.savefig(cm_filt_png, bbox_inches="tight")
            plt.close()
            print(f"[final eval] Saved filtered confusion PNG (1..{num_classes-1}) -> {cm_filt_png}")
        else:
            print("[final eval] No non-zero expression labels present to produce filtered FER report.")

    H, W = X_all.shape[1], X_all.shape[2]
    if landmarks_to_pixels is not None:
        true_px = landmarks_to_pixels(y_lm_all, H=H, W=W)
        pred_px = landmarks_to_pixels(pred_lm_all, H=H, W=W)
    else:
        true_px = np.asarray(y_lm_all).reshape(y_lm_all.shape[0], -1, 2)
        pred_px = np.asarray(pred_lm_all).reshape(pred_lm_all.shape[0], -1, 2)
        true_px[..., 0] *= W; true_px[..., 1] *= H
        pred_px[..., 0] *= W; pred_px[..., 1] *= H

    dists = np.linalg.norm(pred_px - true_px, axis=2)
    per_point_rmse = np.sqrt(np.mean(dists ** 2, axis=0)) if dists.size else np.array([])
    results = {
        "nme": {
            "per_sample_nme": per_sample_nme,
            "per_point_rmse": per_point_rmse,
        },
        "expression": {
            "unweighted_acc": acc_full,
            "weighted_acc": None,
            "confusion_matrix": cm_full,
            "labels": labels_range
        }
    }

    if build_metrics_tables is not None:
        try:
            save_prefix = os.path.join(out_dir, f"{prefix}_metrics")
            _ = build_metrics_tables(results, pred_expr_for_metrics, y_expr_all, expr_sample_weight_test=None,
                                     save_csv_prefix=save_prefix,
                                     save_excel=os.path.join(out_dir, f"{prefix}_metrics.xlsx"),
                                     display_flag=True)
        except Exception:
            print("[final eval] build_metrics_tables failed; skipping.")
    plt.figure(figsize=(6, 5))
    plt.imshow(cm_full, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(f"{prefix} Confusion Matrix (full)")
    plt.colorbar()
    plt.xlabel("Predicted")
    plt.ylabel("True")
    labels = [str(i) for i in labels_range]
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels, rotation=45)
    plt.yticks(tick_marks, labels)
    for i in range(cm_full.shape[0]):
        for j in range(cm_full.shape[1]):
            plt.text(j, i, int(cm_full[i, j]), ha="center", va="center",
                     color="white" if cm_full[i, j] > cm_full.max() / 2.0 else "black")
    plt.tight_layout()
    cm_png = os.path.join(out_dir, f"{prefix}_confusion_full.png")
    plt.savefig(cm_png, bbox_inches="tight")
    plt.close()
    print(f"[final eval] Saved confusion PNG -> {cm_png}")
    num_to_plot = min(max_visualize, X_all.shape[0])
    rand_indices = choose_random_indices(X_all.shape[0], num_to_plot, seed=None)
    saved_paths = plot_landmark_samples(X_all, pred_lm_all, pred_expr_all, y_true_landmarks=y_lm_all, y_true_expr=y_expr_all, nme=per_sample_nme,
                                           indices=rand_indices,
                                           num_samples=num_to_plot, out_dir=viz_dir, show=False)
    print(f"[final eval] Saved {len(saved_paths)} crop-only visualizations -> {viz_dir}")
    pd.DataFrame({"nme": per_sample_nme}).to_csv(os.path.join(out_dir, f"{prefix}_nme.csv"), index=False)
    pd.DataFrame({"true": true_labels_full, "pred": pred_display}).to_csv(os.path.join(out_dir, f"{prefix}_expr_preds_full.csv"), index=False)

    return {
        "mean_nme": mean_nme,
        "median_nme": median_nme,
        "per_sample_nme": per_sample_nme,
        "confusion_full": cm_full,
        "accuracy_full": acc_full,
        "report_full": report_full
    }

def find_csv_logs(root_dir: str) -> List[str]:
    csvs = glob.glob(os.path.join(root_dir, "**", "*.csv"), recursive=True)
    return sorted(csvs)

def safe_legend():
    ax = plt.gca()
    handles, labels = ax.get_legend_handles_labels()
    if labels:
        plt.legend()

def plot_training_csvs(csv_paths: List[str], out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    plot_dir = os.path.join(out_dir, "log_plots")
    os.makedirs(plot_dir, exist_ok=True)
    for p in csv_paths:
        try:
            df = pd.read_csv(p)
        except Exception:
            continue
        if df.shape[0] == 0:
            continue
        name = os.path.splitext(os.path.basename(p))[0]
        plt.figure(figsize=(8, 5))
        if "loss" in df.columns:
            plt.plot(df["loss"], label="loss")
        if "val_loss" in df.columns:
            plt.plot(df["val_loss"], label="val_loss")
        for key in ("landmark_output_loss", "expression_output_loss", "mask_output_loss"):
            if key in df.columns:
                plt.plot(df[key], label=key)
            if f"val_{key}" in df.columns:
                plt.plot(df[f"val_{key}"], label=f"val_{key}")
        plt.title(f"Losses: {name}")
        safe_legend()
        plt.xlabel("epoch"); plt.ylabel("loss"); plt.grid(True)
        outp = os.path.join(plot_dir, f"{name}_losses.png")
        plt.savefig(outp, bbox_inches="tight"); plt.close()
        metric_keys = [c for c in df.columns if any(m in c for m in ["nme", "expr", "iou", "accuracy", "acc"])]
        if metric_keys:
            plt.figure(figsize=(8, 5))
            for k in metric_keys:
                try:
                    plt.plot(df[k], label=k)
                except Exception:
                    pass
                if f"val_{k}" in df.columns:
                    plt.plot(df[f"val_{k}"], label=f"val_{k}")
            plt.title(f"Metrics: {name}")
            safe_legend()
            plt.xlabel("epoch"); plt.grid(True)
            outp2 = os.path.join(plot_dir, f"{name}_metrics.png")
            plt.savefig(outp2, bbox_inches="tight"); plt.close()
    print(f"[plot_training_csvs] Saved plots to {plot_dir}")

def main(args):
    out_run = _ensure_out_run_dir(args.out_dir or config.OUT_DIR, run_prefix="run")
    print("Outputs will be written to:", out_run)

    prepared_dir = args.prepared_dir or getattr(config, "DATA_DIR", "data")
    split = args.split or "test"
    json_path = os.path.join(prepared_dir, f"{split}.json")
    entries = None
    if os.path.exists(json_path):
        with open(json_path, "r") as f:
            entries = json.load(f)

    search_root = getattr(config, "OUT_DIR", ".")
    mask_model_path = args.mask_model 
    head_model_path = args.head_model 
    final_model_path = args.final_model or find_stage_model("final", search_root) or find_stage_model(getattr(config, "RUN_NAME", "tfd68_unet"), search_root)

    print("Discovered models:")
    print(" mask:", mask_model_path)
    print(" head:", head_model_path)
    print(" final:", final_model_path)

    out_size = (getattr(config, "OUT_W", 256), getattr(config, "OUT_H", 256))
    batch_size = args.batch_size or getattr(config, "BATCH_SIZE", 8)
    masks_root = os.path.join(prepared_dir, split, "masks")

    mask_ds_test = build_xy_dataset(
        json_path=json_path,
        images_root=prepared_dir,
        masks_root=masks_root,
        out_size=out_size,
        batch_size=batch_size,
        shuffle=False,
        repeat=False,
        x_keys=("image",),
        y_keys=("mask",),
        normalize_landmarks=True
    )

    lm_ds_test = build_xy_dataset(
        json_path=json_path,
        images_root=prepared_dir,
        masks_root=masks_root,
        out_size=out_size,
        batch_size=batch_size,
        shuffle=False,
        repeat=False,
        x_keys=("image",),
        y_keys=("landmark", "expression"),
        normalize_landmarks=True
    )

    if mask_model_path:
        mask_model = load_model_with_customs(mask_model_path)
        mask_out_dir = os.path.join(out_run, "mask_eval")
        os.makedirs(mask_out_dir, exist_ok=True)
        mask_ious = evaluate_mask_model_stream(mask_model, mask_ds_test, out_dir=mask_out_dir,
                                              prefix="mask_test", batch_size=batch_size,
                                              thr=args.mask_thr, max_visualize=args.max_visualize,
                                              max_batches=args.max_batches)
    else:
        print("Skipping mask evaluation (model not available).")

    if head_model_path:
        head_model = load_model_with_customs(head_model_path)
        head_out_dir = os.path.join(out_run, "head_eval")
        os.makedirs(head_out_dir, exist_ok=True)
        head_res = evaluate_final_stage(head_model, lm_ds_test, out_dir=head_out_dir,
                                         entries=entries, prefix="final_test",
                                         max_visualize=args.max_visualize, max_batches=args.max_batches)
        if head_res:
            print("Head stage mean NME:", head_res["mean_nme"])
        try:
            nme50 = compute_test_nme_batches(head_model, lm_ds_test, num_batches=50)
            if nme50 is not None:
                mean_nme50, per_sample50 = nme50
                print(f"[head eval] mean NME on up to 50 batches: {mean_nme50:.6f} (N={len(per_sample50)})")
                pd.DataFrame({"nme": per_sample50}).to_csv(os.path.join(head_out_dir, "final_test_nme_50batches.csv"), index=False)
                print(f"[head eval] saved 50-batch NME CSV -> {os.path.join(head_out_dir, 'final_test_nme_50batches.csv')}")
        except Exception as e:
            print("[head eval] compute_test_nme_batches failed:", e)
    else:
        print("Skipping Head evaluation (model not available).")

    if final_model_path:
        final_model = load_model_with_customs(final_model_path)
        final_out_dir = os.path.join(out_run, "final_eval")
        os.makedirs(final_out_dir, exist_ok=True)
        final_res = evaluate_final_stage(final_model, lm_ds_test, out_dir=final_out_dir,
                                         entries=entries, prefix="final_test",
                                         max_visualize=args.max_visualize, max_batches=args.max_batches)
        if final_res:
            print("Final stage mean NME:", final_res["mean_nme"])

        try:
            nme50 = compute_test_nme_batches(final_model, lm_ds_test, num_batches=50)
            if nme50 is not None:
                mean_nme50, per_sample50 = nme50
                print(f"[final eval] mean NME on up to 50 batches: {mean_nme50:.6f} (N={len(per_sample50)})")
                pd.DataFrame({"nme": per_sample50}).to_csv(os.path.join(final_out_dir, "final_test_nme_50batches.csv"), index=False)
                print(f"[final eval] saved 50-batch NME CSV -> {os.path.join(final_out_dir, 'final_test_nme_50batches.csv')}")
        except Exception as e:
            print("[final eval] compute_test_nme_batches failed:", e)
    else:
        print("Skipping final evaluation (model not available).")

    csvs = find_csv_logs(getattr(config, "OUT_DIR", "."))
    if csvs:
        plot_training_csvs(csvs, out_dir=out_run)
    else:
        print("No training CSV logs found; skipping training plots.")

    print("All done. Results ->", out_run)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mask_model", type=str, default=None)
    parser.add_argument("--head_model", type=str, default=None)
    parser.add_argument("--final_model", type=str, default=None)
    parser.add_argument("--prepared_dir", type=str, default=None)
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--out_dir", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=getattr(config, "BATCH_SIZE", 8))
    parser.add_argument("--mask_thr", type=float, default=0.5)
    parser.add_argument("--max_visualize", type=int, default=1000)
    parser.add_argument("--max_batches", type=int, default=None)
    args = parser.parse_args()
    main(args)
