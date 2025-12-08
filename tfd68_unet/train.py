import os
import time
import numpy as np
import tensorflow as tf
import config.config as config
import matplotlib
matplotlib.use("Agg")
import pandas as pd
from typing import Optional
from dataset.datasetloader import (
    build_xy_dataset
)

from model.unet import (
    build_unet_trunk,
    build_mask_model,
    build_final_model,
)

from utils.utils import (
    save_loss_plots,
    print_batch_info,
    compute_nme,
    save_all_loss_plot,
    save_trainval_plot
)

from utils.loss import wing_loss

dev = getattr(config, "DEVICE", None)
if dev is not None:
    dev_str = str(dev).strip()
    dev_up = dev_str.upper()
    if dev_up.startswith("CPU"):
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        print("config.DEVICE=CPU -> forcing CPU execution (CUDA_VISIBLE_DEVICES cleared).")
    else:
        idx = None
        if ":" in dev_str:
            try:
                idx = int(dev_str.split(":", 1)[1])
            except Exception:
                idx = 0
        elif dev_str.isdigit():
            try:
                idx = int(dev_str)
            except Exception:
                idx = 0
        else:
            idx = 0
        if idx is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(idx)
            print(f"Set CUDA_VISIBLE_DEVICES={idx} (from config.DEVICE='{dev_str}').")
else:
    print("No config.DEVICE set -> using default CUDA_VISIBLE_DEVICES.")


def enable_tf_gpu_growth():
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print("Enabled TF GPU memory growth for", len(gpus), "GPU(s):", gpus)
        except Exception as e:
            print("Failed to set memory growth:", e)
    else:
        print("No visible GPUs found (TensorFlow sees 0 physical GPUs).")

enable_tf_gpu_growth()

def compute_dataset_nme(model, ds, num_batches: Optional[int] = 50):
    import numpy as _np

    nmes = []
    if ds is None:
        return float("nan"), _np.array([], dtype=float)

    if num_batches is None:
        it = iter(ds)
    else:
        it = iter(ds.take(num_batches))

    for batch in it:
        if isinstance(batch, tuple) and len(batch) >= 2:
            x_batch = batch[0]
            y_batch = batch[1]
        else:
            continue

        if isinstance(y_batch, dict):
            if 'landmark_output' in y_batch:
                y_lm = y_batch['landmark_output']
            elif 'landmark' in y_batch:
                y_lm = y_batch['landmark']
            else:
                continue
        else:
            y_lm = y_batch

        try:
            preds = model.predict_on_batch(x_batch)
        except Exception:
            preds = model(x_batch, training=False)

        if isinstance(preds, (list, tuple)):
            pred_lm = preds[0]
        elif isinstance(preds, dict):
            pred_lm = preds.get('landmark_output') or list(preds.values())[0]
        else:
            pred_lm = preds

        try:
            pred_lm_np = pred_lm.numpy() if hasattr(pred_lm, "numpy") else _np.asarray(pred_lm)
        except Exception:
            pred_lm_np = _np.asarray(pred_lm)

        try:
            y_lm_np = y_lm.numpy() if hasattr(y_lm, "numpy") else _np.asarray(y_lm)
        except Exception:
            y_lm_np = _np.asarray(y_lm)

        img_for_res = None
        try:
            if isinstance(x_batch, dict):
                img_for_res = x_batch.get("image", list(x_batch.values())[0])
            else:
                img_for_res = x_batch
        except Exception:
            img_for_res = None

        batch_nmes = compute_nme(pred_lm_np, y_lm_np, X_images=img_for_res, normalize_by='interocular')
        batch_nmes = _np.asarray(batch_nmes, dtype=float)
        nmes.extend(batch_nmes.tolist())

    if len(nmes) == 0:
        return float("nan"), _np.array([], dtype=float)
    return float(_np.mean(nmes)), _np.array(nmes, dtype=float)

def main():
    DEBUG = True 

    os.makedirs(config.OUT_DIR, exist_ok=True)

    prepared_dir = getattr(config, "DATA_DIR", "data")

    train_json = os.path.join(prepared_dir, "train.json")
    val_json = os.path.join(prepared_dir, "val.json")

    if not os.path.exists(train_json) or not os.path.exists(val_json):
        raise FileNotFoundError(f"Prepared JSONs not found. Expected at {train_json} and {val_json}. Run prepare_dataset.py first or set DATA_DIR correctly in config.")

    out_size = (getattr(config, "OUT_W", 256), getattr(config, "OUT_H", 256))
    num_expr_classes = int(getattr(config, "NUM_CLASSES", 8))
    expr_num = max(1, num_expr_classes - 1)
    batch_size = int(getattr(config, "BATCH_SIZE", 8))

    if DEBUG:
        print(f"[DEBUG] GLOBAL: num_expr_classes={num_expr_classes}, expr_num (C-1)={expr_num}")

    mask_ds_train = build_xy_dataset(
        json_path=train_json,
        images_root=prepared_dir,
        masks_root=os.path.join(prepared_dir, "train", "masks"),
        out_size=out_size,
        batch_size=batch_size,
        shuffle=True,
        repeat=False,
        x_keys=("image",),
        y_keys=("mask",),
        normalize_landmarks=True,
        attach_expression_weights=False,
        keep_classes=None
    )

    mask_ds_val = build_xy_dataset(
        json_path=val_json,
        images_root=prepared_dir,
        masks_root=os.path.join(prepared_dir, "val", "masks"),
        out_size=out_size,
        batch_size=batch_size,
        shuffle=False,
        repeat=False,
        x_keys=("image",),
        y_keys=("mask",),
        normalize_landmarks=True,
        attach_expression_weights=False,
        keep_classes=None
    )

    lm_ds_train = build_xy_dataset(
        json_path=train_json,
        images_root=prepared_dir,
        masks_root=os.path.join(prepared_dir, "train", "masks"),
        out_size=out_size,
        batch_size=batch_size,
        shuffle=True,
        repeat=False,
        x_keys=("image",),
        y_keys=("landmark", "expression"),
        normalize_landmarks=True,
        attach_expression_weights=True,
        keep_classes=None
    )

    lm_ds_val = build_xy_dataset(
        json_path=val_json,
        images_root=prepared_dir,
        masks_root=os.path.join(prepared_dir, "val", "masks"),
        out_size=out_size,
        batch_size=batch_size,
        shuffle=False,
        repeat=False,
        x_keys=("image",),
        y_keys=("landmark", "expression"),
        normalize_landmarks=True,
        attach_expression_weights=True,
        keep_classes=None
    )

    expr_keep = tuple(range(1, num_expr_classes))
    lm_ds_expr = build_xy_dataset(
        json_path=train_json,
        images_root=prepared_dir,
        masks_root=os.path.join(prepared_dir, "train", "masks"),
        out_size=out_size,
        batch_size=batch_size,
        shuffle=True,
        repeat=False,
        x_keys=("image",),
        y_keys=("landmark", "expression"),
        normalize_landmarks=True,
        attach_expression_weights=False,
        keep_classes=expr_keep
    )

    lm_ds_expr_val = build_xy_dataset(
        json_path=val_json,
        images_root=prepared_dir,
        masks_root=os.path.join(prepared_dir, "val", "masks"),
        out_size=out_size,
        batch_size=batch_size,
        shuffle=False,
        repeat=False,
        x_keys=("image",),
        y_keys=("landmark", "expression"),
        normalize_landmarks=True,
        attach_expression_weights=False,
        keep_classes=expr_keep
    )

    if DEBUG:
        print(f"[DEBUG] expr_keep (dataset kept classes) = {expr_keep}")
        try:
            it = iter(lm_ds_expr)
            sample = next(it)
            if isinstance(sample, tuple) and len(sample) >= 2:
                x_s, y_s = sample[0], sample[1]
                if isinstance(y_s, dict) and 'expression_output' in y_s:
                    y_expr_raw = y_s['expression_output']
                else:
                    y_expr_raw = y_s if not isinstance(y_s, dict) else list(y_s.values())[0]
                y_np = y_expr_raw.numpy() if isinstance(y_expr_raw, tf.Tensor) else np.asarray(y_expr_raw)
                print(f"[DEBUG] sample lm_ds_expr y_expr_raw (first batch): shape={y_np.shape}")
                if y_np.ndim == 1:
                    print("[DEBUG] sample labels (sparse):", y_np[:min(8, y_np.size)])
                elif y_np.ndim == 2:
                    print("[DEBUG] sample labels (one-hot) argmax:", np.argmax(y_np, axis=1)[:min(8, y_np.shape[0])])
                else:
                    print("[DEBUG] sample y_expr_raw (unknown rank):", y_np.shape)
        except Exception as e:
            print("[DEBUG] Could not sample from lm_ds_expr:", e)

    def _remap_expr_to_sparse(x, y):
        """
        Dataset map function: return (x, y_sparse) where y_sparse is int32 in 0..expr_num-1.
        Handles batched tensors.
        """
        e = y['expression_output']
        erank = tf.rank(e)
        def _sparse_case():
            e_s = tf.cast(tf.reshape(e, [-1]) - 1, tf.int32)
            return e_s
        def _onehot_case():
            a = tf.argmax(e, axis=-1)
            return tf.cast(a - 1, tf.int32)

        y_sparse = tf.cond(tf.equal(erank, 1), _sparse_case, _onehot_case)
        return x, y_sparse

    expr_train_ds = lm_ds_expr.map(lambda x, y: _remap_expr_to_sparse(x, y), num_parallel_calls=tf.data.AUTOTUNE)
    expr_val_ds = lm_ds_expr_val.map(lambda x, y: _remap_expr_to_sparse(x, y), num_parallel_calls=tf.data.AUTOTUNE)

    if DEBUG:
        try:
            it2 = iter(expr_train_ds)
            x_r, y_r = next(it2)
            y_r_np = y_r.numpy() if isinstance(y_r, tf.Tensor) else np.asarray(y_r)
            print(f"[DEBUG] sample expr_train_ds (after remap) y_sparse shape={y_r_np.shape}; examples: {y_r_np[:min(12, y_r_np.size)]}")
            if y_r_np.size > 0:
                print(f"[DEBUG] remapped label min/max = {y_r_np.min()}/{y_r_np.max()}")
        except Exception as e:
            print("[DEBUG] Could not sample from expr_train_ds (remapped):", e)

    for item in mask_ds_train.take(1):
        print_batch_info(item[0], item[1], prefix="mask_train:")

    sample_lm = next(iter(lm_ds_train.take(1)))
    if len(sample_lm) == 3:
        x_b, y_b, sw_b = sample_lm
        print("lm_train: x type:", type(x_b))
        if isinstance(x_b, dict):
            print("lm_train x keys:", list(x_b.keys()))
        print("lm_train y keys:", list(y_b.keys()))
        if isinstance(sw_b, dict):
            print("lm_train sample_weight keys:", list(sw_b.keys()))
        else:
            print("lm_train sample_weight type:", type(sw_b))
    else:
        x_b = sample_lm[0]
        y_b = sample_lm[1]
        print("Unexpected lm_ds_train element length:", len(sample_lm))
    if isinstance(x_b, dict):
        x_img = x_b.get("image", list(x_b.values())[0])
    else:
        x_img = x_b

    shape_list = x_img.shape.as_list()
    if len(shape_list) < 4:
        raise RuntimeError(f"Unexpected image tensor rank: {shape_list}")
    batch_dim, H, W, C = shape_list[0], shape_list[1], shape_list[2], shape_list[3]
    if H is None or W is None or C is None:
        W_cfg, H_cfg = out_size[0], out_size[1]
        if C is None:
            C = int(tf.shape(x_img)[-1])
        H = H or H_cfg
        W = W or W_cfg

    input_shape = (int(H), int(W), int(C))
    print(f"Derived input_shape from dataset: {input_shape}")

    trunk = build_unet_trunk(input_shape=input_shape, dropout_rate=getattr(config, "DROPOUT", 0.5))

    print("\n=== STAGE 1: Train mask model ===")
    mask_model = build_mask_model(trunk, base_lr=config.BASE_LR)

    callbacks_mask = [
        tf.keras.callbacks.ModelCheckpoint(os.path.join(config.OUT_DIR, "mask_ckpt.h5"), monitor='val_loss', save_best_only=True, verbose=1),
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=config.PATIENCE_ES, restore_best_weights=True, verbose=1),
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=config.PATIENCE_LR, min_lr=config.MIN_LR, verbose=1),
        tf.keras.callbacks.CSVLogger(os.path.join(config.OUT_DIR, "mask_csvlog.csv")),
        tf.keras.callbacks.TensorBoard(log_dir=os.path.join(config.OUT_DIR, "mask_tb"), histogram_freq=0)
    ]
    mask_model.summary()
    history_mask = mask_model.fit(x=mask_ds_train, validation_data=mask_ds_val, epochs=config.EPOCHS_STAGE1, callbacks=callbacks_mask, verbose=1)
    save_loss_plots(history_mask, os.path.join(config.OUT_DIR, "mask_plots"))

    print("\n=== STAGE 2: Freeze trunk ===")
    for layer in trunk.layers:
        layer.trainable = False

    final_model = build_final_model(trunk,
                                    num_landmarks=getattr(config, "KEYPOINTS", 68),
                                    num_classes=expr_num,  
                                    base_lr=config.BASE_LR,
                                    loss_type='sparse_categorical_crossentropy',
                                    use_cbam=False,
                                    compile_model=False)
    final_model.summary()

    epochs = int(getattr(config, "EPOCHS_STAGE3", 10))
    steps_per_epoch = 1000
    val_steps = 50  
    expr_loss_weight = float(getattr(config, "EXPR_LOSS_WEIGHT", 0.5))
    learning_rate = float(getattr(config, "BASE_LR", 1e-4))
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    # Loss functions
    landmark_loss_fn = wing_loss 
    expr_loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False, reduction=tf.keras.losses.Reduction.NONE)

    # Metrics
    train_lm_loss = tf.keras.metrics.Mean(name='train_landmark_loss')
    train_expr_loss = tf.keras.metrics.Mean(name='train_expr_loss')
    train_total_loss = tf.keras.metrics.Mean(name='train_total_loss')
    train_expr_acc = tf.keras.metrics.SparseCategoricalAccuracy(name='train_expr_acc')

    val_lm_loss = tf.keras.metrics.Mean(name='val_landmark_loss')
    val_expr_loss = tf.keras.metrics.Mean(name='val_expr_loss')
    val_total_loss = tf.keras.metrics.Mean(name='val_total_loss')
    val_expr_acc = tf.keras.metrics.SparseCategoricalAccuracy(name='val_expr_acc')
    lm_iter = iter(lm_ds_train.repeat())
    expr_iter = iter(expr_train_ds.repeat())

    lm_val_iter = iter(lm_ds_val.repeat()) if lm_ds_val is not None else None
    expr_val_iter = iter(expr_val_ds.repeat()) if expr_val_ds is not None else None
    best_val = float("inf")
    patience = int(getattr(config, "PATIENCE_ES", 10))
    wait = 0
    ckpt_dir = os.path.join(config.OUT_DIR, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_path = os.path.join(ckpt_dir, "head_customloop_best.h5")

    stage2_train_total = []
    stage2_val_total = []
    stage2_train_lm = []
    stage2_val_lm = []
    stage2_train_expr = []
    stage2_val_expr = []

    print(f"Custom training: epochs={epochs}, steps_per_epoch={steps_per_epoch}, expr_loss_weight={expr_loss_weight}")

    for epoch in range(epochs):
        epoch_start = time.time()
        train_lm_loss.reset_states(); train_expr_loss.reset_states(); train_total_loss.reset_states(); train_expr_acc.reset_states()

        for step in range(steps_per_epoch):
            try:
                lm_elem = next(lm_iter)
            except StopIteration:
                lm_iter = iter(lm_ds_train.repeat()); lm_elem = next(lm_iter)

            try:
                expr_elem = next(expr_iter)
            except StopIteration:
                expr_iter = iter(expr_train_ds.repeat()); expr_elem = next(expr_iter)

            if isinstance(lm_elem, tuple) and len(lm_elem) == 3:
                x_lm, y_lm_dict, sw_lm = lm_elem
            else:
                x_lm, y_lm_dict = lm_elem

            if isinstance(expr_elem, tuple) and len(expr_elem) == 3:
                x_expr, y_expr_dict, sw_expr = expr_elem
                if isinstance(y_expr_dict, dict) and 'expression_output' in y_expr_dict:
                    y_expr_true = y_expr_dict['expression_output']
                else:
                    y_expr_true = y_expr_dict
            else:
                x_expr, y_expr_true = expr_elem

            y_lm_true = y_lm_dict['landmark_output']
            y_expr_sparse = tf.reshape(y_expr_true, [-1])
            with tf.GradientTape() as tape:
                preds_lm_from_lm, preds_expr_from_lm = final_model(x_lm, training=True)
                lm_loss_per_sample = landmark_loss_fn(y_lm_true, preds_lm_from_lm)
                lm_loss = tf.reduce_mean(lm_loss_per_sample) if tf.rank(lm_loss_per_sample) > 0 else lm_loss_per_sample

                preds_lm_from_expr, preds_expr_from_expr = final_model(x_expr, training=True)
                expr_loss_per_sample = expr_loss_fn(y_expr_sparse, preds_expr_from_expr)
                expr_loss = tf.reduce_mean(expr_loss_per_sample)

                total_loss = lm_loss + expr_loss_weight * expr_loss

            grads = tape.gradient(total_loss, final_model.trainable_variables)
            optimizer.apply_gradients(zip(grads, final_model.trainable_variables))

            train_lm_loss.update_state(lm_loss)
            train_expr_loss.update_state(expr_loss)
            train_total_loss.update_state(total_loss)
            train_expr_acc.update_state(y_expr_sparse, preds_expr_from_expr)

            if DEBUG and epoch == 0 and step == 0:
                try:
                    y_sample = y_expr_sparse.numpy()
                    pred_sample_shape = preds_expr_from_expr.shape
                    print(f"[DEBUG][TRAIN] epoch={epoch} step={step} y_expr_sparse.shape={y_sample.shape} num_examples={min(8,y_sample.size)}")
                    print("[DEBUG][TRAIN] y_expr_sparse examples:", y_sample[:min(8, y_sample.size)])
                    print(f"[DEBUG][TRAIN] preds_expr_from_expr.shape = {pred_sample_shape}")
                    if len(pred_sample_shape) > 1 and pred_sample_shape[-1] > 1:
                        p_np = preds_expr_from_expr.numpy()
                        argm = np.argmax(p_np, axis=1)
                        conf = np.max(p_np, axis=1)
                        print("[DEBUG][TRAIN] preds argmax (first 8):", argm[:min(8, argm.size)])
                        print("[DEBUG][TRAIN] preds conf (first 8):", conf[:min(8, conf.size)])
                except Exception as e:
                    print("[DEBUG] Could not print train debug info:", e)

            if (step + 1) % 200 == 0 or (step + 1) == steps_per_epoch:
                print(f"[Epoch {epoch+1}/{epochs}] Step {step+1}/{steps_per_epoch} "
                      f"lm_loss={train_lm_loss.result().numpy():.4f} expr_loss={train_expr_loss.result().numpy():.4f} "
                      f"expr_acc={train_expr_acc.result().numpy():.4f}")

        if lm_val_iter is not None and expr_val_iter is not None:
            val_lm_loss.reset_states(); val_expr_loss.reset_states(); val_total_loss.reset_states(); val_expr_acc.reset_states()
            for vstep in range(val_steps):
                try:
                    v_lm = next(lm_val_iter)
                except StopIteration:
                    lm_val_iter = iter(lm_ds_val.repeat()); v_lm = next(lm_val_iter)
                try:
                    v_expr = next(expr_val_iter)
                except StopIteration:
                    expr_val_iter = iter(expr_val_ds.repeat()); v_expr = next(expr_val_iter)

                if isinstance(v_lm, tuple) and len(v_lm) == 3:
                    x_v_lm, y_v_lm, _ = v_lm
                else:
                    x_v_lm, y_v_lm = v_lm

                if isinstance(v_expr, tuple) and len(v_expr) == 3:
                    x_v_expr, y_v_expr_dict, _ = v_expr
                    if isinstance(y_v_expr_dict, dict) and 'expression_output' in y_v_expr_dict:
                        y_v_expr_true = y_v_expr_dict['expression_output']
                    else:
                        y_v_expr_true = y_v_expr_dict
                else:
                    x_v_expr, y_v_expr_true = v_expr

                y_v_lm_true = y_v_lm['landmark_output']
                y_v_expr_sparse = tf.reshape(y_v_expr_true, [-1])

                preds_v_lm, _ = final_model(x_v_lm, training=False)
                _, preds_v_expr = final_model(x_v_expr, training=False)

                v_lm_loss_val = tf.reduce_mean(landmark_loss_fn(y_v_lm_true, preds_v_lm))
                v_expr_loss_val = tf.reduce_mean(expr_loss_fn(y_v_expr_sparse, preds_v_expr))

                val_lm_loss.update_state(v_lm_loss_val)
                val_expr_loss.update_state(v_expr_loss_val)
                val_total_loss.update_state(v_lm_loss_val + expr_loss_weight * v_expr_loss_val)
                val_expr_acc.update_state(y_v_expr_sparse, preds_v_expr)

                if DEBUG and vstep == 0:
                    try:
                        y_v_np = y_v_expr_sparse.numpy()
                        p_v_np = preds_v_expr.numpy()
                        print(f"[DEBUG][VAL] vstep={vstep} y_v_expr_sparse.shape={y_v_np.shape} examples={y_v_np[:min(8, y_v_np.size)]}")
                        if p_v_np.ndim == 2:
                            print("[DEBUG][VAL] preds_v_expr argmax (first 8):", np.argmax(p_v_np, axis=1)[:min(8, p_v_np.shape[0])])
                            print("[DEBUG][VAL] preds_v_expr conf (first 8):", np.max(p_v_np, axis=1)[:min(8, p_v_np.shape[0])])
                        else:
                            print("[DEBUG][VAL] preds_v_expr (sparse?) shape:", p_v_np.shape)
                    except Exception as e:
                        print("[DEBUG] Could not print val debug info:", e)

            val_total = val_total_loss.result().numpy()
            print(f"Epoch {epoch+1} finished in {time.time() - epoch_start:.1f}s "
                  f"train_lm_loss={train_lm_loss.result().numpy():.4f} train_expr_loss={train_expr_loss.result().numpy():.4f} "
                  f"train_expr_acc={train_expr_acc.result().numpy():.4f}")
            print(f"  VAL lm_loss={val_lm_loss.result().numpy():.4f} val_expr_loss={val_expr_loss.result().numpy():.4f} val_expr_acc={val_expr_acc.result().numpy():.4f} val_total={val_total:.4f}")
            stage2_train_total.append(float(train_total_loss.result().numpy()))
            stage2_train_lm.append(float(train_lm_loss.result().numpy()))
            stage2_train_expr.append(float(train_expr_loss.result().numpy()))
            if lm_val_iter is not None and expr_val_iter is not None:
                stage2_val_total.append(float(val_total))
                stage2_val_lm.append(float(val_lm_loss.result().numpy()))
                stage2_val_expr.append(float(val_expr_loss.result().numpy()))
            else:
                stage2_val_total.append(np.nan)
                stage2_val_lm.append(np.nan)
                stage2_val_expr.append(np.nan)

            if val_total < best_val - 1e-8:
                best_val = val_total
                wait = 0
                try:
                    final_model.save(ckpt_path)
                    print(f"Saved improved checkpoint to {ckpt_path} (val_total={best_val:.6f})")
                except Exception as e:
                    print("Warning: failed to save checkpoint:", e)
            else:
                wait += 1
                print(f"No improvement in val_total (wait={wait}/{patience})")

            if wait >= patience:
                print(f"Early stopping triggered (no improvement for {patience} epochs).")
                break
        else:
            print(f"Epoch {epoch+1} finished in {time.time() - epoch_start:.1f}s "
                  f"train_lm_loss={train_lm_loss.result().numpy():.4f} train_expr_loss={train_expr_loss.result().numpy():.4f} "
                  f"train_expr_acc={train_expr_acc.result().numpy():.4f}")

    try:
        full_plot = save_trainval_plot("stage2", config.OUT_DIR,
                                        stage2_train_total, stage2_val_total,
                                        stage2_train_lm, stage2_val_lm,
                                        stage2_train_expr, stage2_val_expr)
        print("[plot] Stage 2 full loss plot saved ->", full_plot)
    except Exception as e:
        print("[plot] Stage 2 plotting failed:", e)
    final_out = os.path.join(config.OUT_DIR, f"{getattr(config,'RUN_NAME','tfd68_unet')}_head.h5")
    try:
        final_model.save(final_out)
        print("Saved final model to", final_out)
    except Exception as e:
        print("Warning: failed to save final model:", e)

    print("\n=== STAGE 3: Unfreeze trunk & fine-tune full model ===")
    for layer in trunk.layers:
        layer.trainable = True

    final_model.summary()
    lm_iter = iter(lm_ds_train.repeat())
    expr_iter = iter(expr_train_ds.repeat())
    lm_val_iter = iter(lm_ds_val.repeat()) if lm_ds_val is not None else None
    expr_val_iter = iter(expr_val_ds.repeat()) if expr_val_ds is not None else None
    best_val = float("inf")
    patience = int(getattr(config, "PATIENCE_ES", 10))
    wait = 0
    ckpt_dir = os.path.join(config.OUT_DIR, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_path = os.path.join(ckpt_dir, "final_customloop_best.h5")

    stage3_train_total = []
    stage3_val_total = []
    stage3_train_lm = []
    stage3_val_lm = []
    stage3_train_expr = []
    stage3_val_expr = []
    stage3_train_nme = []
    stage3_val_nme = []

    print(f"Custom training: epochs={epochs}, steps_per_epoch={steps_per_epoch}, expr_loss_weight={expr_loss_weight}")

    for epoch in range(epochs):
        epoch_start = time.time()
        train_lm_loss.reset_states(); train_expr_loss.reset_states(); train_total_loss.reset_states(); train_expr_acc.reset_states()

        for step in range(steps_per_epoch):
            try:
                lm_elem = next(lm_iter)
            except StopIteration:
                lm_iter = iter(lm_ds_train.repeat()); lm_elem = next(lm_iter)

            try:
                expr_elem = next(expr_iter)
            except StopIteration:
                expr_iter = iter(expr_train_ds.repeat()); expr_elem = next(expr_iter)

            if isinstance(lm_elem, tuple) and len(lm_elem) == 3:
                x_lm, y_lm_dict, sw_lm = lm_elem
            else:
                x_lm, y_lm_dict = lm_elem

            if isinstance(expr_elem, tuple) and len(expr_elem) == 3:
                x_expr, y_expr_dict, sw_expr = expr_elem
                if isinstance(y_expr_dict, dict) and 'expression_output' in y_expr_dict:
                    y_expr_true = y_expr_dict['expression_output']
                else:
                    y_expr_true = y_expr_dict
            else:
                x_expr, y_expr_true = expr_elem

            y_lm_true = y_lm_dict['landmark_output']

            y_expr_sparse = tf.reshape(y_expr_true, [-1])

            with tf.GradientTape() as tape:
                preds_lm_from_lm, preds_expr_from_lm = final_model(x_lm, training=True)
                lm_loss_per_sample = landmark_loss_fn(y_lm_true, preds_lm_from_lm)
                lm_loss = tf.reduce_mean(lm_loss_per_sample) if tf.rank(lm_loss_per_sample) > 0 else lm_loss_per_sample

                preds_lm_from_expr, preds_expr_from_expr = final_model(x_expr, training=True)
                expr_loss_per_sample = expr_loss_fn(y_expr_sparse, preds_expr_from_expr)
                expr_loss = tf.reduce_mean(expr_loss_per_sample)

                total_loss = lm_loss + expr_loss_weight * expr_loss

            grads = tape.gradient(total_loss, final_model.trainable_variables)
            optimizer.apply_gradients(zip(grads, final_model.trainable_variables))

            train_lm_loss.update_state(lm_loss)
            train_expr_loss.update_state(expr_loss)
            train_total_loss.update_state(total_loss)
            train_expr_acc.update_state(y_expr_sparse, preds_expr_from_expr)

            if DEBUG and epoch == 0 and step == 0:
                try:
                    y_sample = y_expr_sparse.numpy()
                    pred_sample_shape = preds_expr_from_expr.shape
                    print(f"[DEBUG][TRAIN] epoch={epoch} step={step} y_expr_sparse.shape={y_sample.shape} num_examples={min(8,y_sample.size)}")
                    print("[DEBUG][TRAIN] y_expr_sparse examples:", y_sample[:min(8, y_sample.size)])
                    print(f"[DEBUG][TRAIN] preds_expr_from_expr.shape = {pred_sample_shape}")
                    if len(pred_sample_shape) > 1 and pred_sample_shape[-1] > 1:
                        p_np = preds_expr_from_expr.numpy()
                        argm = np.argmax(p_np, axis=1)
                        conf = np.max(p_np, axis=1)
                        print("[DEBUG][TRAIN] preds argmax (first 8):", argm[:min(8, argm.size)])
                        print("[DEBUG][TRAIN] preds conf (first 8):", conf[:min(8, conf.size)])
                except Exception as e:
                    print("[DEBUG] Could not print train debug info:", e)

            # logging
            if (step + 1) % 200 == 0 or (step + 1) == steps_per_epoch:
                print(f"[Epoch {epoch+1}/{epochs}] Step {step+1}/{steps_per_epoch} "
                      f"lm_loss={train_lm_loss.result().numpy():.4f} expr_loss={train_expr_loss.result().numpy():.4f} "
                      f"expr_acc={train_expr_acc.result().numpy():.4f}")

        if lm_val_iter is not None and expr_val_iter is not None:
            val_lm_loss.reset_states(); val_expr_loss.reset_states(); val_total_loss.reset_states(); val_expr_acc.reset_states()
            for vstep in range(val_steps):
                try:
                    v_lm = next(lm_val_iter)
                except StopIteration:
                    lm_val_iter = iter(lm_ds_val.repeat()); v_lm = next(lm_val_iter)
                try:
                    v_expr = next(expr_val_iter)
                except StopIteration:
                    expr_val_iter = iter(expr_val_ds.repeat()); v_expr = next(expr_val_iter)

                if isinstance(v_lm, tuple) and len(v_lm) == 3:
                    x_v_lm, y_v_lm, _ = v_lm
                else:
                    x_v_lm, y_v_lm = v_lm

                if isinstance(v_expr, tuple) and len(v_expr) == 3:
                    x_v_expr, y_v_expr_dict, _ = v_expr
                    if isinstance(y_v_expr_dict, dict) and 'expression_output' in y_v_expr_dict:
                        y_v_expr_true = y_v_expr_dict['expression_output']
                    else:
                        y_v_expr_true = y_v_expr_dict
                else:
                    x_v_expr, y_v_expr_true = v_expr

                y_v_lm_true = y_v_lm['landmark_output']
                y_v_expr_sparse = tf.reshape(y_v_expr_true, [-1])

                preds_v_lm, _ = final_model(x_v_lm, training=False)
                _, preds_v_expr = final_model(x_v_expr, training=False)

                v_lm_loss_val = tf.reduce_mean(landmark_loss_fn(y_v_lm_true, preds_v_lm))
                v_expr_loss_val = tf.reduce_mean(expr_loss_fn(y_v_expr_sparse, preds_v_expr))

                val_lm_loss.update_state(v_lm_loss_val)
                val_expr_loss.update_state(v_expr_loss_val)
                val_total_loss.update_state(v_lm_loss_val + expr_loss_weight * v_expr_loss_val)
                val_expr_acc.update_state(y_v_expr_sparse, preds_v_expr)

                if DEBUG and vstep == 0:
                    try:
                        y_v_np = y_v_expr_sparse.numpy()
                        p_v_np = preds_v_expr.numpy()
                        print(f"[DEBUG][VAL] vstep={vstep} y_v_expr_sparse.shape={y_v_np.shape} examples={y_v_np[:min(8, y_v_np.size)]}")
                        if p_v_np.ndim == 2:
                            print("[DEBUG][VAL] preds_v_expr argmax (first 8):", np.argmax(p_v_np, axis=1)[:min(8, p_v_np.shape[0])])
                            print("[DEBUG][VAL] preds_v_expr conf (first 8):", np.max(p_v_np, axis=1)[:min(8, p_v_np.shape[0])])
                        else:
                            print("[DEBUG][VAL] preds_v_expr (sparse?) shape:", p_v_np.shape)
                    except Exception as e:
                        print("[DEBUG] Could not print val debug info:", e)

            val_total = val_total_loss.result().numpy()
            print(f"Epoch {epoch+1} finished in {time.time() - epoch_start:.1f}s "
                  f"train_lm_loss={train_lm_loss.result().numpy():.4f} train_expr_loss={train_expr_loss.result().numpy():.4f} "
                  f"train_expr_acc={train_expr_acc.result().numpy():.4f}")
            print(f"  VAL lm_loss={val_lm_loss.result().numpy():.4f} val_expr_loss={val_expr_loss.result().numpy():.4f} val_expr_acc={val_expr_acc.result().numpy():.4f} val_total={val_total:.4f}")
            stage3_train_total.append(float(train_total_loss.result().numpy()))
            stage3_train_lm.append(float(train_lm_loss.result().numpy()))
            stage3_train_expr.append(float(train_expr_loss.result().numpy()))
            if lm_val_iter is not None and expr_val_iter is not None:
                stage3_val_total.append(float(val_total_loss.result().numpy()))
                stage3_val_lm.append(float(val_lm_loss.result().numpy()))
                stage3_val_expr.append(float(val_expr_loss.result().numpy()))
            else:
                stage3_val_total.append(np.nan)
                stage3_val_lm.append(np.nan)
                stage3_val_expr.append(np.nan)

            try:
                train_nme_val, train_nme_arr = compute_dataset_nme(final_model, lm_ds_train, num_batches=None)
                val_nme_val, val_nme_arr = compute_dataset_nme(final_model, lm_ds_val, num_batches=None)

                stage3_train_nme.append(float(train_nme_val))
                stage3_val_nme.append(float(val_nme_val))

                try:
                    import pandas as _pd
                    _pd.DataFrame({"train_nme": [train_nme_val], "val_nme": [val_nme_val]}).to_csv(
                        os.path.join(config.OUT_DIR, "stage3_nme_epoch_history.csv"), mode='a', index=False, header=not os.path.exists(os.path.join(config.OUT_DIR, "stage3_nme_epoch_history.csv"))
                    )
                except Exception:
                    pass

                print(f"[stage3 NME] epoch={epoch+1} train_nme={train_nme_val:.6f} val_nme={val_nme_val:.6f}")
            except Exception as e:
                print("[stage3 NME] failed to compute NME this epoch:", e)
                
            if val_total < best_val - 1e-8:
                best_val = val_total
                wait = 0
                try:
                    final_model.save(ckpt_path)
                    print(f"Saved improved checkpoint to {ckpt_path} (val_total={best_val:.6f})")
                except Exception as e:
                    print("Warning: failed to save checkpoint:", e)
            else:
                wait += 1
                print(f"No improvement in val_total (wait={wait}/{patience})")

            if wait >= patience:
                print(f"Early stopping triggered (no improvement for {patience} epochs).")
                break
        else:
            print(f"Epoch {epoch+1} finished in {time.time() - epoch_start:.1f}s "
                  f"train_lm_loss={train_lm_loss.result().numpy():.4f} train_expr_loss={train_expr_loss.result().numpy():.4f} "
                  f"train_expr_acc={train_expr_acc.result().numpy():.4f}")
            
    try:
        full3 = save_trainval_plot("stage3", config.OUT_DIR,
                                    stage3_train_total, stage3_val_total,
                                    stage3_train_lm, stage3_val_lm,
                                    stage3_train_expr, stage3_val_expr)
        print("[plot] Stage 3 full loss plot saved ->", full3)
        total_only3 = save_all_loss_plot("stage3", config.OUT_DIR, stage3_train_total, stage3_val_total)
        print("[plot] Stage 3 total-only loss plot saved ->", total_only3)
    except Exception as e:
        print("[plot] Stage 3 plotting failed:", e)

    try:
        pd.DataFrame({
            "epoch": list(range(1, len(stage3_train_nme) + 1)),
            "train_nme": stage3_train_nme,
            "val_nme": stage3_val_nme
        }).to_csv(os.path.join(config.OUT_DIR, "stage3_nme_history.csv"), index=False)
        print("[plot] Stage 3 NME history saved ->", os.path.join(config.OUT_DIR, "stage3_nme_history.csv"))
    except Exception:
        pass

    final_out = os.path.join(config.OUT_DIR, f"{getattr(config,'RUN_NAME','tfd68_unet')}_full.h5")
    try:
        final_model.save(final_out)
        print("Saved final model to", final_out)
    except Exception as e:
        print("Warning: failed to save final model:", e)

    print("Training complete.")


if __name__ == "__main__":
    main()
