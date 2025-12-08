import os
import datetime
from typing import Optional, Dict, Any, List, Tuple
import numpy as np
import cv2
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight
from matplotlib.lines import Line2D
import config.config as config

def make_run_paths(out_dir: str, run_name: str, ts: Optional[str] = None) -> Dict[str, str]:
    if ts is None:
        ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    ckpt_path = os.path.join(out_dir, f"{run_name}_best.h5")
    csv_log = os.path.join(out_dir, f"{run_name}_history.csv")
    tb_logdir = os.path.join(out_dir, "tb", f"{run_name}_{ts}")
    return {"ckpt_path": ckpt_path, "csv_log": csv_log, "tb_logdir": tb_logdir}


def make_paths_for_runs(out_dir: str, run_names: Dict[str, str], ts: Optional[str] = None) -> Dict[str, Dict[str, str]]:
    return {k: make_run_paths(out_dir, rn, ts=ts) for k, rn in run_names.items()}


def save_loss_plots(history, out_prefix: str):
    hist = history.history
    # Loss plot
    plt.figure(figsize=(8, 5))
    if 'loss' in hist:
        plt.plot(hist['loss'], label='loss')
    if 'val_loss' in hist:
        plt.plot(hist['val_loss'], label='val_loss')
    plt.xlabel('epoch'); plt.ylabel('loss'); plt.title('Loss'); plt.legend()
    loss_path = f"{out_prefix}_loss.png"
    plt.tight_layout()
    plt.savefig(loss_path)
    plt.close()
    metric_keys = [k for k in hist.keys() if k not in ('loss', 'val_loss')]
    if not metric_keys:
        return
    plt.figure(figsize=(10, 6))
    for k in metric_keys:
        plt.plot(hist[k], label=k)
    plt.xlabel('epoch'); plt.ylabel('value'); plt.title('Metrics'); plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    met_path = f"{out_prefix}_metrics.png"
    plt.tight_layout()
    plt.savefig(met_path)
    plt.close()
    print("Saved plots:", loss_path, met_path)
    
def plot_loss_curves(history, out_path: Optional[str] = None, show: bool = True):
    hist = getattr(history, "history", history)
    plt.figure(figsize=(10, 6))
    # total loss
    if 'loss' in hist:
        plt.plot(hist['loss'], label='Train Total Loss')
    if 'val_loss' in hist:
        plt.plot(hist['val_loss'], label='Val Total Loss')

    if 'landmark_output_loss' in hist:
        plt.plot(hist['landmark_output_loss'], label='Train Landmark Loss')
    if 'val_landmark_output_loss' in hist:
        plt.plot(hist['val_landmark_output_loss'], label='Val Landmark Loss')

    if 'expression_output_loss' in hist:
        plt.plot(hist['expression_output_loss'], label='Train Expression Loss')
    if 'val_expression_output_loss' in hist:
        plt.plot(hist['val_expression_output_loss'], label='Val Expression Loss')

    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training & Validation Loss Curves')
    plt.legend()
    plt.grid(True)
    if out_path:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        plt.savefig(out_path, bbox_inches='tight')
    if show:
        plt.show()
    plt.close()


def preds_to_pixel_coords(lm_flat: np.ndarray, H: int, W: int) -> Tuple[np.ndarray, np.ndarray]:
    lm = np.array(lm_flat).reshape(-1, 2)
    if lm.size == 0:
        return np.array([], dtype=int), np.array([], dtype=int)
    if lm.max() <= 1.0 + 1e-6:
        xs = (lm[:, 0] * W).astype(np.int32)
        ys = (lm[:, 1] * H).astype(np.int32)
        return xs, ys

    c0 = lm[:, 0]
    c1 = lm[:, 1]
    if np.all((c0 >= 0) & (c0 <= W) & (c1 >= 0) & (c1 <= H)):
        xs = np.clip(c0, 0, W - 1).astype(np.int32)
        ys = np.clip(c1, 0, H - 1).astype(np.int32)
        return xs, ys
    if np.all((c0 >= 0) & (c0 <= H) & (c1 >= 0) & (c1 <= W)):
        ys = np.clip(c0, 0, H - 1).astype(np.int32)
        xs = np.clip(c1, 0, W - 1).astype(np.int32)
        return xs, ys
    xs = np.clip(c0, 0, W - 1).astype(np.int32)
    ys = np.clip(c1, 0, H - 1).astype(np.int32)
    return xs, ys


def plot_landmark_samples(X: np.ndarray,
                          y_pred_landmarks: np.ndarray,
                          y_pred_expr: np.ndarray,
                          y_true_landmarks: Optional[np.ndarray] = None,
                          y_true_expr: Optional[np.ndarray] = None,
                          nme: Optional[np.ndarray] = None,
                          indices: Optional[List[int]] = None,
                          num_samples: int = 6,
                          out_dir: Optional[str] = None,
                          show: bool = False):
    if X is None or X.shape[0] == 0:
        return []
    N = X.shape[0]
    H = X.shape[1]
    W = X.shape[2]
    if indices is None:
        indices = np.random.choice(N, min(num_samples, N), replace=False).tolist()

    pred_P = None
    if isinstance(y_pred_expr, np.ndarray) and y_pred_expr.ndim == 2:
        pred_P = y_pred_expr.shape[1]
    num_expr_classes = int(getattr(config, "NUM_CLASSES", 8))

    saved_paths = []
    for idx in indices:
        crop = X[idx]
        if crop.ndim == 3 and crop.shape[2] >= 1:
            im_gray = crop[:, :, 0]
        else:
            im_gray = crop
        if im_gray.max() <= 1.01:
            img_vis = (im_gray * 255.0).astype(np.uint8)
        else:
            img_vis = im_gray.astype(np.uint8)
        img_vis = cv2.cvtColor(img_vis, cv2.COLOR_GRAY2RGB)

        lm_pred = np.asarray(y_pred_landmarks[idx]).reshape(-1)
        xs_p, ys_p = preds_to_pixel_coords(lm_pred, H, W)
        for x, y in zip(xs_p, ys_p):
            cv2.circle(img_vis, (int(x), int(y)), radius=2, color=(0, 255, 0), thickness=-1)

        if y_true_landmarks is not None:
            lm_true = np.asarray(y_true_landmarks[idx]).reshape(-1)
            xs_t, ys_t = preds_to_pixel_coords(lm_true, H, W)
            for x, y in zip(xs_t, ys_t):
                cv2.circle(img_vis, (int(x), int(y)), radius=2, color=(255, 0, 0), thickness=-1)

        probs = np.asarray(y_pred_expr[idx])
        if probs.ndim == 1:
            if probs.size > 1:
                pred_internal = int(np.argmax(probs))
                pred_conf = float(np.max(probs))
            else:
                pred_internal = int(probs.reshape(-1)[0])
                pred_conf = 1.0
        else:
            arr = probs.reshape(-1)
            if arr.size > 1:
                pred_internal = int(np.argmax(arr))
                pred_conf = float(np.max(arr))
            else:
                pred_internal = int(arr[0])
                pred_conf = 1.0

        if pred_P is not None and pred_P == num_expr_classes - 1:
            pred_display = pred_internal + 1
        else:
            pred_display = pred_internal

        actual_expr = None
        if y_true_expr is not None:
            y = y_true_expr[idx]
            y = np.asarray(y)
            if y.ndim == 1 and y.size > 1:
                actual_expr_val = int(np.argmax(y))
            else:
                actual_expr_val = int(y.reshape(-1)[0])
            actual_expr = actual_expr_val

        nme_str = ""
        if nme is not None:
            try:
                nme_val = float(np.asarray(nme).reshape(-1)[idx])
                nme_str = f" NME: {nme_val:.4f}"
            except Exception:
                nme_str = ""

        if actual_expr == 0:
            title = f""
            if nme_str:
                title += f"{nme_str}"
        else:
            title = f"Expr: {pred_display} ({pred_conf:.2f})"
            if actual_expr is not None:
                title += f"   Actual: {actual_expr}"
            if nme_str:
                title += f"{nme_str}"

        plt.figure(figsize=(4, 4))
        plt.imshow(img_vis)
        plt.title(title)
        plt.axis('off')

        legend_items = []
        legend_items.append(Line2D([], [], color=(0, 1, 0), marker='o', linestyle='None',
                          markersize=6, label='Predicted landmarks'))
        if y_true_landmarks is not None:
            legend_items.append(Line2D([], [], color=(1, 0, 0), marker='o', linestyle='None',
                            markersize=6, label='Ground-truth landmarks'))
        if legend_items:
            plt.legend(handles=legend_items, loc='upper right', fontsize='x-small')

        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
            p = os.path.join(out_dir, f"sample_{idx:04d}.png")
            plt.savefig(p, bbox_inches='tight')
            saved_paths.append(p)
        if show:
            plt.show()
        plt.close()

    return saved_paths

RIGHT_EYE_IDX = list(range(36, 42))
LEFT_EYE_IDX  = list(range(42, 48))

def _ensure_landmark_array(lm_arr: np.ndarray) -> np.ndarray:
    a = np.asarray(lm_arr)
    if a.ndim == 1:
        K = a.shape[0] // 2
        return a.reshape(1, K, 2).astype(np.float32)
    if a.ndim == 2:
        N, C = a.shape
        if C % 2 != 0:
            raise ValueError("Landmark flatten length must be even")
        K = C // 2
        return a.reshape(N, K, 2).astype(np.float32)
    if a.ndim == 3:
        return a.astype(np.float32)
    raise ValueError("Unsupported landmarks array ndim=%d" % a.ndim)


def landmarks_to_pixels(lm_preds: np.ndarray, H: Optional[int] = None, W: Optional[int] = None) -> np.ndarray:
    arr = _ensure_landmark_array(lm_preds)
    N, K, _ = arr.shape

    if arr.max() <= 1.0 + 1e-6:
        if H is None or W is None:
            raise ValueError("Landmarks appear normalized but H/W not provided.")
        out = np.zeros_like(arr, dtype=np.float32)
        out[:, :, 0] = arr[:, :, 0] * W  # x
        out[:, :, 1] = arr[:, :, 1] * H  # y
        return out

    out = np.zeros_like(arr, dtype=np.float32)
    for i in range(N):
        lm = arr[i]
        c0 = lm[:, 0]
        c1 = lm[:, 1]
        # detect ordering heuristics
        if H is not None and W is not None and np.all((c0 >= 0) & (c0 <= W) & (c1 >= 0) & (c1 <= H)):
            xs = c0; ys = c1
        elif H is not None and W is not None and np.all((c0 >= 0) & (c0 <= H) & (c1 >= 0) & (c1 <= W)):
            ys = c0; xs = c1
        else:
            if (c0.max() - c0.min()) > (c1.max() - c1.min()):
                xs = c0; ys = c1
            else:
                xs = c1; ys = c0
        out[i, :, 0] = xs
        out[i, :, 1] = ys
    return out


def compute_nme(y_pred_landmarks: np.ndarray,
                y_true_landmarks: np.ndarray,
                X_images: Optional[np.ndarray] = None,
                normalize_by: str = 'interocular',
                print_summary: bool = True) -> Dict[str, Any]:
    H = X_images.shape[1] if X_images is not None else None
    W = X_images.shape[2] if X_images is not None else None

    pred_pix = landmarks_to_pixels(y_pred_landmarks, H=H, W=W)
    true_pix = landmarks_to_pixels(y_true_landmarks, H=H, W=W)

    N = pred_pix.shape[0]
    K = pred_pix.shape[1]

    per_sample_nme = np.zeros((N,), dtype=np.float32)
    per_point_rmse_accum = np.zeros((K,), dtype=np.float64)

    for i in range(N):
        p = pred_pix[i]
        t = true_pix[i]

        if normalize_by == 'interocular':
            if K > 45:
                right_corner = t[36]
                left_corner = t[45]
                norm_d = np.linalg.norm(right_corner - left_corner)
            elif K >= 48:
                right_center = t[36:42].mean(axis=0)
                left_center = t[42:48].mean(axis=0)
                norm_d = np.linalg.norm(right_center - left_center)
            else:
                xmin, ymin = t[:, 0].min(), t[:, 1].min()
                xmax, ymax = t[:, 0].max(), t[:, 1].max()
                norm_d = np.sqrt((xmax - xmin) ** 2 + (ymax - ymin) ** 2)
        elif normalize_by == 'bbox':
            xmin, ymin = t[:, 0].min(), t[:, 1].min()
            xmax, ymax = t[:, 0].max(), t[:, 1].max()
            norm_d = np.sqrt((xmax - xmin) ** 2 + (ymax - ymin) ** 2)
        else:
            raise ValueError("Unknown normalize_by: %s" % normalize_by)

        if norm_d < 1e-6:
            xmin, ymin = t[:, 0].min(), t[:, 1].min()
            xmax, ymax = t[:, 0].max(), t[:, 1].max()
            norm_d = np.sqrt((xmax - xmin) ** 2 + (ymax - ymin) ** 2)
            if norm_d < 1e-6:
                norm_d = 1.0 

        dists = np.linalg.norm(p - t, axis=1)
        per_sample_nme[i] = dists.mean() / (norm_d + 1e-12)
        per_point_rmse_accum += dists ** 2

    per_point_rmse = np.sqrt(per_point_rmse_accum / max(1, N))
    mean_nme = float(per_sample_nme.mean())
    median_nme = float(np.median(per_sample_nme))

    if print_summary:
        print("NME (mean): {:.6f} (median: {:.6f})".format(mean_nme, median_nme))
        print("Per-point RMSE (first 10):", per_point_rmse[:10])

    return {
        "mean_nme": mean_nme,
        "median_nme": median_nme,
        "per_sample_nme": per_sample_nme,
        "per_point_rmse": per_point_rmse,
        "pred_pixels": pred_pix,
        "true_pixels": true_pix
    }


def to_label_array(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y)
    if y.ndim == 2:
        return np.argmax(y, axis=1)
    elif y.ndim == 1:
        return y.astype(int)
    else:
        raise ValueError("Unsupported label shape: %s" % (y.shape,))


def compute_expression_accuracy(y_pred_expr: np.ndarray,
                                y_true_expr: np.ndarray,
                                sample_weights: Optional[np.ndarray] = None,
                                print_report: bool = True) -> Dict[str, Any]:
    y_pred_labels = to_label_array(y_pred_expr)
    y_true_labels = to_label_array(y_true_expr)
    N = y_true_labels.shape[0]
    correct = (y_pred_labels == y_true_labels).astype(np.float32)
    unweighted_acc = float(correct.mean())

    weighted_acc = None
    if sample_weights is not None:
        sw = np.asarray(sample_weights).astype(np.float32)
        if sw.shape[0] != N:
            raise ValueError("sample_weights length mismatch")
        denom = sw.sum()
        weighted_acc = float((sw * correct).sum() / (denom + 1e-12))

    labels = np.unique(np.concatenate([y_true_labels, y_pred_labels]))
    labels_sorted = np.sort(labels)
    label_to_idx = {int(l): i for i, l in enumerate(labels_sorted)}
    C = len(labels_sorted)
    conf = np.zeros((C, C), dtype=np.int32)
    for t, p in zip(y_true_labels, y_pred_labels):
        conf[label_to_idx[int(t)], label_to_idx[int(p)]] += 1

    per_class_acc = {}
    for i, lab in enumerate(labels_sorted):
        row = conf[i]
        total = row.sum()
        per_class_acc[int(lab)] = float(row[i] / total) if total > 0 else None

    if print_report:
        print("Expression accuracy (unweighted): {:.4f}".format(unweighted_acc))
        if weighted_acc is not None:
            print("Expression accuracy (weighted):   {:.4f}".format(weighted_acc))
        print("Per-class accuracy:")
        for lab in labels_sorted:
            print("  class {:2d}: {}".format(int(lab), ("{:.4f}".format(per_class_acc[int(lab)]) if per_class_acc[int(lab)] is not None else "N/A")))
        print("Confusion matrix (rows=true, cols=pred) labels:", labels_sorted)
        print(conf)

    return {
        "unweighted_acc": unweighted_acc,
        "weighted_acc": weighted_acc,
        "per_class_acc": per_class_acc,
        "confusion_matrix": conf,
        "labels": labels_sorted
    }

def evaluate_test_set(X_test: np.ndarray,
                      y_pred_landmarks: np.ndarray,
                      y_pred_expr: np.ndarray,
                      y_expr_test: np.ndarray,
                      y_true_landmarks: np.ndarray,
                      expr_sample_weight_test: Optional[np.ndarray] = None,
                      normalize_by: str = 'interocular',
                      print_summary: bool = True) -> Dict[str, Any]:
    nme_res = compute_nme(y_pred_landmarks, y_true_landmarks, X_images=X_test, normalize_by=normalize_by, print_summary=print_summary)
    expr_res = compute_expression_accuracy(y_pred_expr, y_expr_test, sample_weights=expr_sample_weight_test, print_report=print_summary)
    return {"nme": nme_res, "expression": expr_res}


def build_metrics_tables(results: Dict[str, Any],
                         y_pred_expr: np.ndarray,
                         y_true_expr: np.ndarray,
                         expr_sample_weight_test: Optional[np.ndarray] = None,
                         save_csv_prefix: str = "metrics",
                         save_excel: str = "metrics_summary.xlsx",
                         display_flag: bool = True) -> Dict[str, pd.DataFrame]:
    mean_nme = float(np.mean(results['nme']['per_sample_nme']))
    median_nme = float(np.median(results['nme']['per_sample_nme']))
    per_point_rmse = np.array(results['nme']['per_point_rmse'])
    mean_rmse = float(np.mean(per_point_rmse)) if per_point_rmse.size else 0.0
    expr_unweighted = float(results['expression']['unweighted_acc'])
    expr_weighted = results['expression'].get('weighted_acc', None)

    summary = {
        "mean_nme": [mean_nme],
        "median_nme": [median_nme],
        "mean_landmark_rmse": [mean_rmse],
        "expr_unweighted_acc": [expr_unweighted],
        "expr_weighted_acc": [expr_weighted if expr_weighted is not None else np.nan],
        "num_samples": [len(results['nme']['per_sample_nme'])]
    }
    df_summary = pd.DataFrame(summary)

    per_sample_nme = np.array(results['nme']['per_sample_nme']).reshape(-1)
    N = per_sample_nme.shape[0]

    y_pred_expr_arr = np.asarray(y_pred_expr)
    if y_pred_expr_arr.ndim == 2:
        pred_labels = np.argmax(y_pred_expr_arr, axis=1)
        pred_conf = np.max(y_pred_expr_arr, axis=1)
    else:
        pred_labels = y_pred_expr_arr.astype(int)
        pred_conf = np.ones_like(pred_labels, dtype=float)

    y_true_expr_arr = np.asarray(y_true_expr)
    if y_true_expr_arr.ndim == 2:
        true_labels = np.argmax(y_true_expr_arr, axis=1)
    else:
        true_labels = y_true_expr_arr.astype(int)

    data = {
        "index": np.arange(N),
        "nme": per_sample_nme,
        "pred_expr": pred_labels,
        "true_expr": true_labels,
        "pred_conf": pred_conf
    }
    if expr_sample_weight_test is not None:
        sw = np.asarray(expr_sample_weight_test).astype(float)
        if sw.shape[0] != N:
            if sw.shape[0] > N:
                sw = sw[:N]
            else:
                sw = np.concatenate([sw, np.ones((N-sw.shape[0],), dtype=float)], axis=0)
        data["sample_weight"] = sw
    df_per_sample = pd.DataFrame(data)

    df_per_point = pd.DataFrame({
        "landmark_idx": np.arange(len(per_point_rmse)),
        "rmse": per_point_rmse
    })

    conf = results['expression']['confusion_matrix']
    labels = results['expression']['labels']
    df_conf = pd.DataFrame(conf, index=[f"true_{int(l)}" for l in labels],
                           columns=[f"pred_{int(l)}" for l in labels])

    if display_flag:
        print("\n=== Summary ===")
        display_df = df_summary.copy()
        display_df["mean_nme"] = display_df["mean_nme"].map(lambda v: f"{v:.6f}")
        display_df["median_nme"] = display_df["median_nme"].map(lambda v: f"{v:.6f}")
        display_df["mean_landmark_rmse"] = display_df["mean_landmark_rmse"].map(lambda v: f"{v:.4f}")
        print(display_df.to_string(index=False))
        print("\n=== Per-sample (first 20) ===")
        print(df_per_sample.head(20).to_string(index=False))
        print("\n=== Per-landmark RMSE (first 10) ===")
        print(df_per_point.head(10).to_string(index=False))
        print("\n=== Confusion matrix ===")
        print(df_conf)

    df_summary.to_csv(f"{save_csv_prefix}_summary.csv", index=False)
    df_per_sample.to_csv(f"{save_csv_prefix}_per_sample.csv", index=False)
    df_per_point.to_csv(f"{save_csv_prefix}_per_point_rmse.csv", index=False)
    df_conf.to_csv(f"{save_csv_prefix}_confusion_matrix.csv")

    try:
        with pd.ExcelWriter(save_excel) as writer:
            df_summary.to_excel(writer, sheet_name="summary", index=False)
            df_per_sample.to_excel(writer, sheet_name="per_sample", index=False)
            df_per_point.to_excel(writer, sheet_name="per_landmark_rmse", index=False)
            df_conf.to_excel(writer, sheet_name="confusion_matrix", index=False)
    except Exception as e:
        print("Warning: failed to write Excel:", e)

    print(f"\nCSV files saved: {save_csv_prefix}_*.csv")
    print(f"Excel workbook saved (if possible): {save_excel}")

    return {
        "df_summary": df_summary,
        "df_per_sample": df_per_sample,
        "df_per_point": df_per_point,
        "df_confusion": df_conf
    }

def inspect_array(name: str, a: Any, max_show: int = 20):
    a = np.asarray(a)
    print(f"\n--- {name} ---")
    print("shape:", a.shape, "dtype:", a.dtype)
    try:
        print("min/max:", float(a.min()), float(a.max()))
    except Exception:
        pass
    if a.ndim == 2:
        row_sums = a.sum(axis=1)
        print("row-sum min/max:", float(row_sums.min()), float(row_sums.max()))
    flat = a.flatten()
    uniq = np.unique(flat)
    print("unique values (sample up to 20):", uniq[:max_show])
    print("example rows (first 5):")
    if a.ndim == 1:
        print(flat[:10])
    else:
        for i in range(min(5, a.shape[0])):
            print(a[i])
    return


def denormalize_labels_if_needed(y, C: int):
    y = np.asarray(y)
    if y.ndim == 2 and y.shape[1] == C:
        return np.argmax(y, axis=1)
    if y.ndim == 2 and y.shape[1] == 1:
        y1 = y[:, 0]
        if (y1.dtype.kind in 'f') and (y1.max() <= 1.0 + 1e-6):
            labels = np.round(y1 * (C - 1)).astype(int)
            return labels.clip(0, C-1)
        else:
            return y1.astype(int).clip(0, C-1)
    if y.ndim == 1:
        if y.dtype.kind in 'f' and y.max() <= 1.0 + 1e-6 and not np.allclose(y, np.round(y)):
            labels = np.round(y * (C - 1)).astype(int)
            return labels.clip(0, C-1)
        else:
            return y.astype(int).clip(0, C-1)
    if y.ndim > 1:
        return np.argmax(y, axis=1)
    raise ValueError("Unhandled label shape: %s" % (y.shape,))


def filter_out_class0_and_remap(ds, old_num_classes: int, new_num_classes: int, batch_size: int):
    ds_un = ds.unbatch()

    def keep_not_class0(img, y):
        expr_onehot = y['expression_output']
        expr_idx = tf.argmax(expr_onehot, axis=-1, output_type=tf.int32)
        return tf.not_equal(expr_idx, 0)

    ds_filt = ds_un.filter(keep_not_class0)

    def remap_fn(img, y):
        expr_onehot = y['expression_output']
        expr_idx = tf.argmax(expr_onehot, axis=-1, output_type=tf.int32)
        expr_idx_shifted = expr_idx - 1
        expr_onehot_new = tf.one_hot(expr_idx_shifted, depth=new_num_classes, dtype=tf.float32)
        y_new = {
            "landmark_output": y["landmark_output"],
            "expression_output": expr_onehot_new,
            "occlusion_output": y["occlusion_output"]
        }
        return img, y_new

    ds_mapped = ds_filt.map(remap_fn, num_parallel_calls=tf.data.AUTOTUNE)
    ds_batched = ds_mapped.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds_batched

def binarize_mask_dataset(ds, threshold: float = 0.5):
    def map_fn(x, y):
        y_bin = tf.cast(tf.greater(y, threshold), tf.float32)
        return x, y_bin
    return ds.map(map_fn, num_parallel_calls=tf.data.AUTOTUNE)

def convert_occlusion_vector_to_scalar_on_batched_ds(ds: tf.data.Dataset) -> tf.data.Dataset:
    spec = ds.element_spec
    if not isinstance(spec, tuple):
        raise ValueError("Expected dataset element_spec to be a tuple (img, y) or (img, y, sw). Got: %r" % (spec,))

    if len(spec) not in (2, 3):
        raise ValueError("Expected dataset to yield 2 or 3 elements, found %d" % len(spec))

    y_spec = spec[1]
    if not (isinstance(y_spec, dict) and 'occlusion_output' in y_spec):
        raise ValueError("Dataset y element_spec must be a dict containing key 'occlusion_output'. Got: %r" % (y_spec,))
    
    def _map_no_sw(img, y):
        occl_vec = y['occlusion_output']
        occl_scalar = tf.cast(tf.reduce_max(occl_vec, axis=-1, keepdims=True), tf.float32) 
        y_new = {
            "landmark_output": y["landmark_output"],
            "expression_output": y["expression_output"],
            "occlusion_output": occl_scalar
        }
        return img, y_new

    def _map_with_sw(img, y, sw):
        occl_vec = y['occlusion_output']
        occl_scalar = tf.cast(tf.reduce_max(occl_vec, axis=-1, keepdims=True), tf.float32)
        y_new = {
            "landmark_output": y["landmark_output"],
            "expression_output": y["expression_output"],
            "occlusion_output": occl_scalar
        }
        return img, y_new, sw

    if len(spec) == 3:
        return ds.map(_map_with_sw, num_parallel_calls=tf.data.AUTOTUNE)
    else:
        return ds.map(_map_no_sw, num_parallel_calls=tf.data.AUTOTUNE)
    
def denormalize_from_crop(pred_norm_flat, rec, crop_size):
    new_w = rec['width']
    new_h = rec['height']
    bx0, by0, bx1, by1 = rec['orig_bbox']
    crop_w = (bx1 - bx0) + 1
    crop_h = (by1 - by0) + 1
    scale_crop_x = float(new_w) / float(crop_w)
    scale_crop_y = float(new_h) / float(crop_h)

    flat = np.asarray(pred_norm_flat).reshape(-1)
    xs_norm = flat[0::2]
    ys_norm = flat[1::2]
    xs_px_on_saved = xs_norm * new_w
    ys_px_on_saved = ys_norm * new_h
    xs_on_orig = (xs_px_on_saved / scale_crop_x) + bx0
    ys_on_orig = (ys_px_on_saved / scale_crop_y) + by0
    return np.stack([xs_on_orig, ys_on_orig], axis=1)

def compute_expression_sample_weights(lm_ds, num_classes: int):
    all_labels = []
    for batch in lm_ds:
        _, y = batch
        if isinstance(y, dict):
            expr = y.get("expression_output", None)
            if expr is None:
                raise RuntimeError("No 'expression_output' key found in dataset y dict.")
        else:
            expr = y
        expr_np = np.array(expr.numpy(), dtype=np.int32)
        all_labels.append(expr_np)
    labels = np.concatenate(all_labels, axis=0)
    classes = np.arange(num_classes)
    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=labels
    )
    class_weight_map = {int(cls): float(w) for cls, w in zip(classes, class_weights)}
    expr_sample_weight = np.array([class_weight_map[int(l)] for l in labels], dtype=np.float32)
    expr_sample_weight /= (expr_sample_weight.mean() + 1e-8)

    return class_weight_map, expr_sample_weight, labels

def attach_expression_only_weights(ds: tf.data.Dataset,
                                   weights: np.ndarray,
                                   batch_size: int,
                                   expression_output_name: str = "expression_output",
                                   landmark_output_name: str = "landmark_output",
                                   landmark_weight_value: float = 1.0,
                                   shuffle: bool = False,
                                   repeat: bool = False) -> tf.data.Dataset:
    weight_ds = tf.data.Dataset.from_tensor_slices(weights.astype('float32'))

    if shuffle:
        weight_ds = weight_ds.shuffle(buffer_size=len(weights), reshuffle_each_iteration=True)
    if repeat:
        weight_ds = weight_ds.repeat()
    weight_ds = weight_ds.batch(batch_size, drop_remainder=False)
    zipped = tf.data.Dataset.zip((ds, weight_ds))

    def map_fn(xy, w_batch):
        x, y = xy 
        w_batch = tf.cast(w_batch, tf.float32)
        sample_weight = {
            expression_output_name: w_batch,
            landmark_output_name: tf.fill(tf.shape(w_batch), tf.cast(landmark_weight_value, tf.float32))
        }
        return x, y, sample_weight

    return zipped.map(map_fn, num_parallel_calls=tf.data.AUTOTUNE)
def print_batch_info(x_batch, y_batch, prefix=""):
    if isinstance(x_batch, dict):
        for k, v in x_batch.items():
            print(f"{prefix} x['{k}'] shape: {v.shape}")
    else:
        print(f"{prefix} x shape: {x_batch.shape}")

    if isinstance(y_batch, dict):
        for k, v in y_batch.items():
            print(f"{prefix} y['{k}'] shape: {v.shape} dtype: {v.dtype}")
    else:
        print(f"{prefix} y shape: {y_batch.shape} dtype: {y_batch.dtype}")

        
def filter_out_class0_xy(x, y):
    if isinstance(y, dict):
        expr = y.get("expression_output")
    else:
        expr = y[1]
    return tf.not_equal(tf.squeeze(expr), tf.constant(0, dtype=tf.int32))


def save_all_loss_plot(stage_name: str, out_dir: str, train_total: list, val_total: list):
    os.makedirs(out_dir, exist_ok=True)
    epochs = list(range(1, len(train_total) + 1))
    tt = np.array(train_total, dtype=float)
    vt = np.array(val_total, dtype=float) if val_total is not None else None

    plt.figure(figsize=(7, 4))
    plt.plot(epochs, tt, label="total_train", marker='o')
    if vt is not None:
        plt.plot(epochs, vt, label="total_val", marker='o')
    plt.title(f"{stage_name} total loss")
    plt.xlabel("epoch"); plt.ylabel("loss")
    plt.grid(True); plt.legend(loc="best", fontsize="small")
    outp = os.path.join(out_dir, f"{stage_name}_all_losses.png")
    try:
        plt.tight_layout(); plt.savefig(outp, bbox_inches="tight")
    except Exception as e:
        print(f"[plot] failed to save all loss plot for {stage_name}: {e}")
    plt.close()
    return outp

def save_trainval_plot(stage_name: str,
                           out_dir: str,
                           train_total: list,
                           val_total: list,
                           train_lm: list,
                           val_lm: list,
                           train_expr: list,
                           val_expr: list):

    os.makedirs(out_dir, exist_ok=True)
    epochs = list(range(1, len(train_total) + 1))

    tt = np.array(train_total, dtype=float)
    vt = np.array(val_total, dtype=float) if val_total is not None else None
    tl = np.array(train_lm, dtype=float)
    vl = np.array(val_lm, dtype=float) if val_lm is not None else None
    te = np.array(train_expr, dtype=float)
    ve = np.array(val_expr, dtype=float) if val_expr is not None else None

    plt.figure(figsize=(9, 6))
    plt.plot(epochs, tt, label="total_train", marker='o')
    if vt is not None:
        plt.plot(epochs, vt, label="total_val", marker='o')
    plt.plot(epochs, tl, label="landmark_train", marker='x')
    if vl is not None:
        plt.plot(epochs, vl, label="landmark_val", marker='x')
    plt.plot(epochs, te, label="expr_train", marker='s')
    if ve is not None:
        plt.plot(epochs, ve, label="expr_val", marker='s')

    plt.title(f"{stage_name} losses")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.grid(True)
    plt.legend(loc="best", fontsize="small")
    full_path = os.path.join(out_dir, f"{stage_name}_losses_full.png")
    try:
        plt.tight_layout()
        plt.savefig(full_path, bbox_inches="tight")
    except Exception as e:
        print(f"[plot] failed to save full loss plot for {stage_name}: {e}")
    plt.close()

    return full_path