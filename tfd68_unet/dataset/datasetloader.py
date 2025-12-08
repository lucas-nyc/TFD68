
from __future__ import annotations
import os
import json
import logging
import re
from typing import List, Dict, Any, Optional, Tuple, Iterator
import numpy as np
import cv2
from tqdm import tqdm
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight
import random
try:
    import tfd68_unet.config.config as config
except Exception:
    import config.config as config

logger = logging.getLogger("tfd68_dataloader")
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

THERMAL_RE = re.compile(
    r"thermal_(?P<id>\d+)_(?P<yaw>-?\d+)_(?P<pitch>-?\d+)(?:\.\w+)?$",
    re.IGNORECASE
)

def split_annotations(out_json: str,
                      out_dir: str,
                      splits: Dict[str, float] = None,
                      seed: Optional[int] = None,
                      save_split_files: bool = True,
                      force_test_ids: Optional[List[Any]] = None
                      ) -> Dict[str, List[Dict[str, Any]]]:
    splits = splits or getattr(config, "SPLIT", {"train": 0.7, "val": 0.15, "test": 0.15})
    seed = seed if seed is not None else getattr(config, "RANDOM_SEED", 42)

    if force_test_ids is None:
        force_test_ids = getattr(config, "FORCE_TEST_IDS", None)

    forced_set = set()
    if force_test_ids:
        for f in force_test_ids:
            try:
                forced_set.add(str(int(f)))
            except Exception:
                forced_set.add(str(f))

    with open(out_json, "r") as f:
        entries = json.load(f)

    id_to_entries: Dict[str, List[Dict[str, Any]]] = {}
    for e in entries:
        fname = e.get("file_name") or e.get("path") or ""
        token = get_id_from_fname_safe(fname)
        key = str(token) if token is not None else str(e.get("image_id"))
        id_to_entries.setdefault(key, []).append(e)

    unique_ids = list(id_to_entries.keys())

    missing_forced = [fid for fid in forced_set if fid not in id_to_entries]
    if missing_forced:
        logger.warning("Forced test IDs not found in annotations and will be ignored: %s", missing_forced)
        forced_set = {fid for fid in forced_set if fid in id_to_entries}

    rng = np.random.RandomState(seed)

    available_ids = [uid for uid in unique_ids if uid not in forced_set]
    rng.shuffle(available_ids)

    n = len(available_ids)
    train_frac = splits.get("train", 0.7)
    val_frac = splits.get("val", 0.15)
    test_frac = splits.get("test", max(0.0, 1.0 - train_frac - val_frac))

    n_train = int(round(train_frac * n))
    n_val = int(round(val_frac * n))

    train_ids = available_ids[:n_train]
    val_ids = available_ids[n_train:n_train + n_val]
    test_ids = available_ids[n_train + n_val:]

    for fid in forced_set:
        if fid not in test_ids:
            test_ids.append(fid)

    out = {"train": [], "val": [], "test": []}
    for _id in train_ids:
        out["train"].extend(id_to_entries[_id])
    for _id in val_ids:
        out["val"].extend(id_to_entries[_id])
    for _id in test_ids:
        out["test"].extend(id_to_entries[_id])

    if save_split_files:
        os.makedirs(out_dir, exist_ok=True)
        for name in ("train", "val", "test"):
            p = os.path.join(out_dir, f"cleaned_{name}.json")
            with open(p, "w") as f:
                json.dump(out[name], f, indent=2)
            logger.info("Wrote %d entries to %s", len(out[name]), p)

    logger.info("Split summary: train_ids=%d, val_ids=%d, test_ids=%d (forced_test_ids=%s)",
                len(train_ids), len(val_ids), len(test_ids), sorted(list(forced_set)))

    return out

def get_id_from_fname_safe(fname: str) -> Optional[str]:
    if not fname:
        return None
    base = os.path.basename(fname)
    m = THERMAL_RE.search(base)
    if m:
        return m.group("id")
    parts = base.split("_")
    if len(parts) >= 2 and parts[0].lower().startswith("thermal"):
        try:
            return parts[1]
        except Exception:
            return None
    return None

def should_use_image_for_cat(anns: List[Dict[str, Any]], required_cat: int = config.REQUIRED_CAT) -> bool:
    for ann in anns:
        try:
            if int(ann.get("category_id", -1)) == int(required_cat):
                return True
        except Exception:
            continue
    return False

def build_multi_dataset(entries: List[Dict[str, Any]],
                        images_root: str,
                        masks_root: str,
                        out_size: Tuple[int, int],
                        batch_size: int = 8,
                        shuffle: bool = True,
                        buffer_size: int = 256,
                        flip_mode: str = "none",
                        keypoints: int = None,
                        num_classes: int = None,
                        train_class_weight_map: Dict[int, float] = None,
                        return_sample_weights: bool = False
                        ) -> Tuple[tf.data.Dataset, Dict[int, float]]:
    keypoints = keypoints or config.KEYPOINTS
    num_classes = num_classes or config.NUM_CLASSES
    if out_size is None:
        out_w, out_h = getattr(config, "OUT_W", 256), getattr(config, "OUT_H", 256)
    else:
        out_w, out_h = out_size

    labels = np.array([int(e.get("class", 0)) for e in entries], dtype=int)
    if train_class_weight_map is None:
        classes = np.unique(labels) if labels.size > 0 else np.array([0])
        if classes.size == 0:
            class_weight_map = {}
        else:
            cw = compute_class_weight('balanced', classes=classes, y=labels)
            class_weight_map = {int(c): float(w) for c, w in zip(classes, cw)}
    else:
        class_weight_map = train_class_weight_map

    def gen():
        yield from _sample_generator_from_entries(entries, images_root, masks_root, out_size=(out_w, out_h),
                                                  keypoints=keypoints, flip_mode=flip_mode, num_classes=num_classes,
                                                  gauss_sigma=None, class_weight_map=class_weight_map)

    output_signature = (
        tf.TensorSpec(shape=(out_h, out_w, 1), dtype=tf.float32), 
        tf.TensorSpec(shape=(out_h, out_w, 1), dtype=tf.float32), 
        tf.TensorSpec(shape=(2 * keypoints,), dtype=tf.float32),
        tf.TensorSpec(shape=(keypoints,), dtype=tf.float32),
        tf.TensorSpec(shape=(), dtype=tf.int32),
        tf.TensorSpec(shape=(), dtype=tf.float32) 
    )

    ds_raw = tf.data.Dataset.from_generator(gen, output_signature=output_signature)

    def to_model_inputs(img, mask, lm_flat, occl, expr_label, sw):
        expr_onehot = tf.one_hot(expr_label, depth=num_classes, dtype=tf.float32)
        y = {
            "landmark_output": tf.cast(lm_flat, tf.float32),
            "expression_output": tf.cast(expr_onehot, tf.float32),
            "occlusion_output": tf.cast(occl, tf.float32)
        }
        if return_sample_weights:
            sw_dict = {"expression_output": tf.cast(sw, tf.float32)}
            return img, y, sw_dict
        else:
            return img, y

    ds_mapped = ds_raw.map(to_model_inputs, num_parallel_calls=tf.data.AUTOTUNE)

    if shuffle:
        ds_mapped = ds_mapped.shuffle(buffer_size=buffer_size)

    ds_batched = ds_mapped.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    return ds_batched, class_weight_map

def build_mirror_map_68(K: int) -> List[int]:
    if K == 68:
        m: List[int] = []
        m += list(range(16, -1, -1))
        m += list(range(26, 21, -1))
        m += list(range(21, 16, -1))
        m += [27, 28, 29, 30]
        m += list(range(35, 30, -1))
        m += list(range(47, 41, -1))
        m += list(range(41, 35, -1))
        m += list(range(59, 47, -1))
        m += list(range(67, 59, -1))
        if len(m) != K:
            raise RuntimeError("mirror map length mismatch")
        return m
    else:
        return list(range(K - 1, -1, -1))


def _sample_generator_from_entries(entries: List[Dict[str, Any]],
                                   images_root: str,
                                   masks_root: str,
                                   out_size: Optional[Tuple[int, int]] = None,
                                   keypoints: int = None,
                                   flip_mode: str = "none",
                                   num_classes: Optional[int] = None,
                                   gauss_sigma: float = None,
                                   class_weight_map: Optional[Dict[int, float]] = None
                                   ) -> Iterator[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, float]]:
    if out_size is None:
        out_w, out_h = getattr(config, "OUT_W", 256), getattr(config, "OUT_H", 256)
    else:
        out_w, out_h = out_size

    gauss_sigma = gauss_sigma if gauss_sigma is not None else config.GAUSS_SIGMA
    keypoints = keypoints or config.KEYPOINTS
    mirror_map = build_mirror_map_68(keypoints)
    num_classes = int(num_classes) if num_classes is not None else max(1, getattr(config, "NUM_CLASSES", 2))

    for entry in entries:
        fname = entry.get("file_name") or entry.get("path") or ""
        file_basename = os.path.basename(fname)

        img_path = None
        if entry.get("path") and os.path.isabs(entry.get("path")) and os.path.exists(entry.get("path")):
            img_path = entry.get("path")
        else:
            path_field = entry.get("path")
            if path_field:
                normalized = path_field.replace("\\", "/").lstrip("/\\")
                cand = os.path.normpath(os.path.join(images_root, normalized))
                if os.path.exists(cand):
                    img_path = cand

        if img_path is None:
            candidates = []
            candidates.append(os.path.join(images_root, file_basename))
            if entry.get("file_name"):
                candidates.append(os.path.join(images_root, entry.get("file_name")))
            if os.path.exists(os.path.join(images_root, os.path.basename(file_basename))):
                candidates.append(os.path.join(images_root, os.path.basename(file_basename)))
            _id = get_id_from_fname_safe(file_basename)
            if _id:
                candidates.append(os.path.join(images_root, str(_id), file_basename))
                candidates.append(os.path.join(images_root, "images", str(_id), file_basename))
            for c in candidates:
                if os.path.exists(c):
                    img_path = c
                    break

        if img_path is None:
            logger.debug("Image not found for entry %s, skipping", fname)
            continue

        im = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        if im is None:
            logger.debug("cv2 failed to read %s (skipping)", img_path)
            continue
        if im.ndim == 3:
            im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        else:
            im_gray = im.copy()

        src_h, src_w = im_gray.shape[:2]
        if src_w <= 0 or src_h <= 0:
            logger.debug("Invalid image size for %s, skipping", img_path)
            continue

        im_resized = cv2.resize(im_gray, (out_w, out_h), interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255.0
        img_single = im_resized[..., None].astype(np.float32)
        img = img_single
        mask = None
        mask_full = None
        base_no_ext = os.path.splitext(file_basename)[0]
        mask_candidates = [
            os.path.join(masks_root, base_no_ext + ".png"),
            os.path.join(masks_root, file_basename.replace(".jpg", ".png")),
            os.path.join(masks_root, file_basename.replace(".JPG", ".png"))
        ]
        _id = get_id_from_fname_safe(file_basename)
        if _id:
            mask_candidates.append(os.path.join(masks_root, str(_id), base_no_ext + ".png"))
            mask_candidates.append(os.path.join(masks_root, "images", str(_id), base_no_ext + ".png"))

        for mc in mask_candidates:
            mc_norm = os.path.normpath(mc)
            if os.path.exists(mc_norm):
                mask_full_tmp = cv2.imread(mc_norm, cv2.IMREAD_UNCHANGED)
                if mask_full_tmp is not None:
                    if mask_full_tmp.ndim == 3:
                        mask_full_tmp = cv2.cvtColor(mask_full_tmp, cv2.COLOR_BGR2GRAY)
                    mask_full = mask_full_tmp
                    break

        if mask_full is not None:
            try:
                mask = cv2.resize(mask_full, (out_w, out_h), interpolation=cv2.INTER_NEAREST).astype(np.float32) / 255.0
            except Exception:
                mask = None

        kps_flat = entry.get("keypoints", []) or []
        if not kps_flat:
            continue
        kps = np.array(kps_flat, dtype=float).reshape(-1, 3)
        xs = kps[:, 0].copy()
        ys = kps[:, 1].copy()

        if mask is None:
            combined = np.zeros((out_h, out_w), dtype=np.float32)
            if np.nanmax(xs) <= 1.01 and np.nanmax(ys) <= 1.01:
                xs_abs = xs * out_w
                ys_abs = ys * out_h
            elif (np.nanmax(xs) <= float(out_w) + 1.0) and (np.nanmax(ys) <= float(out_h) + 1.0):
                xs_abs = xs.copy()
                ys_abs = ys.copy()
            elif "orig_bbox" in entry and entry.get("orig_size"):
                bx0, by0, bx1, by1 = entry.get("orig_bbox", [0, 0, src_w - 1, src_h - 1])
                orig_w, orig_h = entry.get("orig_size", [src_w, src_h])
                crop_w = float((bx1 - bx0) + 1)
                crop_h = float((by1 - by0) + 1)
                scale_crop_x = float(out_w) / crop_w if crop_w > 0 else 1.0
                scale_crop_y = float(out_h) / crop_h if crop_h > 0 else 1.0
                xs_abs = (xs - bx0) * scale_crop_x
                ys_abs = (ys - by0) * scale_crop_y
            else:
                xs_abs = xs.copy()
                ys_abs = ys.copy()
            for i_k in range(keypoints):
                xg = xs_abs[i_k]
                yg = ys_abs[i_k]
                vg = int(kps[i_k, 2])
                if vg == 0:
                    continue
                combined += _gaussian_blob(out_h, out_w, xg, yg, sigma=gauss_sigma)
            mask = np.clip(combined, 0.0, 1.0).astype(np.float32)
        if np.nanmax(xs) <= 1.01 and np.nanmax(ys) <= 1.01:
            landmark_norm = np.stack([xs, ys], axis=1)
        elif (np.nanmax(xs) <= float(out_w) + 1.0) and (np.nanmax(ys) <= float(out_h) + 1.0):
            landmark_norm = np.stack([xs / float(out_w), ys / float(out_h)], axis=1)
        elif "orig_bbox" in entry and entry.get("orig_size"):
            bx0, by0, bx1, by1 = entry.get("orig_bbox", [0, 0, src_w - 1, src_h - 1])
            crop_w = float((bx1 - bx0) + 1)
            crop_h = float((by1 - by0) + 1)
            scale_crop_x = float(out_w) / crop_w if crop_w > 0 else 1.0
            scale_crop_y = float(out_h) / crop_h if crop_h > 0 else 1.0
            xs_mapped = (xs - bx0) * scale_crop_x
            ys_mapped = (ys - by0) * scale_crop_y
            landmark_norm = np.stack([xs_mapped / float(out_w), ys_mapped / float(out_h)], axis=1)
        else:
            landmark_norm = np.stack([xs / float(out_w), ys / float(out_h)], axis=1)

        landmark_norm = np.clip(landmark_norm, 0.0, 1.0)

        occl_flags = np.array([1.0 if int(v) == 1 else 0.0 for v in kps[:, 2]], dtype=np.float32)

        lm_flat = landmark_norm.reshape(-1).astype(np.float32)

        expr_label = int(entry.get("class", 0))
        sw = 1.0
        if class_weight_map is not None:
            sw = float(class_weight_map.get(int(expr_label), 1.0))
        yield img.astype(np.float32), mask.astype(np.float32)[..., None], lm_flat, occl_flags, int(expr_label), float(sw)

        if flip_mode == "both" or flip_mode == "horizontal":
            img_flip = cv2.flip(im_resized, 1)[..., None].astype(np.float32)
            mask_flip = cv2.flip((mask * 255.0).astype(np.uint8), 1).astype(np.float32) / 255.0 if mask is not None else np.zeros((out_h, out_w), dtype=np.float32)
            mask_flip = mask_flip[..., None]
            lm_arr2 = landmark_norm.copy()
            lm_flip = lm_arr2.copy()
            lm_flip[:, 0] = 1.0 - lm_flip[:, 0]
            lm_flip_reordered = lm_flip[mirror_map, :].reshape(-1).astype(np.float32)
            occl_flip = np.array([occl_flags[i] for i in mirror_map], dtype=np.float32)

            yield img_flip.astype(np.float32), mask_flip.astype(np.float32), lm_flip_reordered, occl_flip, int(expr_label), float(sw)



def _gaussian_blob(h: int, w: int, cx: float, cy: float, sigma: float) -> np.ndarray:
    if cx < 0 or cy < 0 or cx >= w or cy >= h:
        return np.zeros((h, w), dtype=np.float32)
    xs = np.arange(w, dtype=np.float32)
    ys = np.arange(h, dtype=np.float32)[:, None]
    g = np.exp(-((xs - cx) ** 2 + (ys - cy) ** 2) / (2.0 * sigma ** 2))
    g /= (g.max() + 1e-12)
    return g.astype(np.float32)

def generate_masks(prepared_dir: str,
                   splits: Tuple[str, ...] = ("train", "val", "test"),
                   out_size: Optional[Tuple[int, int]] = None,
                   sigma: Optional[float] = None,
                   mode: str = "disk",
                   radius: Optional[int] = None,
                   overwrite: bool = False,
                   verbose: bool = True) -> None:
    out_size = out_size or (getattr(config, "OUT_W", 128), getattr(config, "OUT_H", 128))
    out_w, out_h = out_size
    default_sigma = float(getattr(config, "GAUSS_SIGMA", 6.0))
    sigma = float(sigma) if sigma is not None else max(1.0, default_sigma / 3.0)

    cfg_radius = getattr(config, "MASK_RADIUS", None)
    if radius is None:
        if cfg_radius is not None:
            radius = cfg_radius
        else:
            radius = max(1, int(min(out_w, out_h) / 64))

    if mode not in ("disk", "gaussian"):
        raise ValueError("mode must be 'disk' or 'gaussian'")

    for split in splits:
        json_path = os.path.join(prepared_dir, f"{split}.json")
        if not os.path.exists(json_path):
            if verbose:
                logger.info("No JSON for split %s at %s - skipping", split, json_path)
            continue

        with open(json_path, "r") as f:
            entries = json.load(f)

        masks_out_dir = os.path.join(prepared_dir, split, "masks")
        os.makedirs(masks_out_dir, exist_ok=True)

        if verbose:
            logger.info("Generating masks for split %s -> %s (N=%d) mode=%s sigma=%.2f radius=%d",
                        split, masks_out_dir, len(entries), mode, sigma, radius)

        for e in tqdm(entries, desc=f"masks:{split}"):
            basename = e.get("file_name") or e.get("path") or ""
            if not basename:
                continue
            mask_name = os.path.splitext(os.path.basename(basename))[0] + ".png"
            mask_out_path = os.path.join(masks_out_dir, mask_name)
            if os.path.exists(mask_out_path) and not overwrite:
                continue

            kps_flat = e.get("keypoints", []) or []
            if not kps_flat:
                continue

            prepared_img_path = os.path.join(prepared_dir, split, "images", os.path.basename(basename))
            if os.path.exists(prepared_img_path):
                im_p = cv2.imread(prepared_img_path, cv2.IMREAD_UNCHANGED)
                if im_p is not None:
                    entry_h, entry_w = im_p.shape[:2]
                else:
                    entry_w = int(e.get("width") or out_w)
                    entry_h = int(e.get("height") or out_h)
            else:
                entry_w = int(e.get("width") or out_w)
                entry_h = int(e.get("height") or out_h)

            if entry_w <= 0 or entry_h <= 0:
                entry_w, entry_h = out_w, out_h

            final_scale_x = float(out_w) / float(entry_w)
            final_scale_y = float(out_h) / float(entry_h)
            
            kps = np.array(kps_flat, dtype=float).reshape(-1, 3)
            xs = kps[:, 0]
            ys = kps[:, 1]

            if np.nanmax(xs) <= 1.01 and np.nanmax(ys) <= 1.01:
                xs_abs = xs * entry_w
                ys_abs = ys * entry_h
            elif (np.nanmax(xs) <= float(entry_w) + 1.0) and (np.nanmax(ys) <= float(entry_h) + 1.0):
                xs_abs = xs.copy()
                ys_abs = ys.copy()
            elif "orig_bbox" in e and e.get("orig_size"):
                bx0, by0, bx1, by1 = e.get("orig_bbox", [0, 0, entry_w - 1, entry_h - 1])
                orig_w, orig_h = e.get("orig_size", [entry_w, entry_h])
                crop_w = float((bx1 - bx0) + 1)
                crop_h = float((by1 - by0) + 1)
                scale_crop_x = float(entry_w) / crop_w if crop_w > 0 else 1.0
                scale_crop_y = float(entry_h) / crop_h if crop_h > 0 else 1.0
                xs_abs = (xs - bx0) * scale_crop_x
                ys_abs = (ys - by0) * scale_crop_y

            else:
                xs_abs = xs.copy()
                ys_abs = ys.copy()

            xs_abs = np.clip(xs_abs, 0.0, float(entry_w) - 1.0)
            ys_abs = np.clip(ys_abs, 0.0, float(entry_h) - 1.0)
            if mode == "disk":
                mask_uint8 = np.zeros((out_h, out_w), dtype=np.uint8)
                for (xg, yg, vg) in zip(xs_abs, ys_abs, kps[:, 2]):
                    if int(vg) == 0:
                        continue
                    x_c = int(round(xg * final_scale_x))
                    y_c = int(round(yg * final_scale_y))
                    if x_c < 0 or y_c < 0 or x_c >= out_w or y_c >= out_h:
                        continue
                    cv2.circle(mask_uint8, (x_c, y_c), int(radius), 255, thickness=-1, lineType=cv2.LINE_AA)
                try:
                    cv2.imwrite(mask_out_path, mask_uint8)
                except Exception as ex:
                    logger.warning("Failed to write mask %s: %s", mask_out_path, ex)
                continue

            combined = np.zeros((out_h, out_w), dtype=np.float32)
            sigma_final = float(sigma)
            for (xg_abs, yg_abs, vg) in zip(xs_abs, ys_abs, kps[:, 2]):
                if int(vg) == 0:
                    continue
                x_c = xg_abs * final_scale_x
                y_c = yg_abs * final_scale_y
                combined += _gaussian_blob(out_h, out_w, x_c, y_c, sigma_final)
            combined = np.clip(combined, 0.0, 1.0)
            mask_uint8 = (combined * 255.0).astype("uint8")
            try:
                cv2.imwrite(mask_out_path, mask_uint8)
            except Exception as ex:
                logger.warning("Failed to write mask %s: %s", mask_out_path, ex)

    if verbose:
        logger.info("Mask generation completed.")

class AnnotationsBuilder:
    def __init__(self,
                 keypoints: int = config.KEYPOINTS,
                 num_classes: int = config.NUM_CLASSES,
                 bbox_margin_px: int = config.BBOX_MARGIN_PX,
                 gauss_sigma: float = config.GAUSS_SIGMA,
                 out_w: int = config.OUT_W,
                 out_h: int = config.OUT_H,
                 required_cat: int = config.REQUIRED_CAT,
                 pitch_map: Dict[str, int] = config.PITCH_TO_EMOTION):
        self.KEYPOINTS = keypoints
        self.NUM_CLASSES = num_classes
        self.BBOX_MARGIN_PX = bbox_margin_px
        self.GAUSS_SIGMA = gauss_sigma
        self.OUT_W = out_w
        self.OUT_H = out_h
        self.REQUIRED_CAT = required_cat
        self.pitch_to_emotion = pitch_map

    def create_standard_annotations(self,
                                    coco_json_path: str,
                                    out_json_path: str,
                                    required_cat: Optional[int] = None,
                                    bbox_margin: Optional[int] = None) -> List[Dict[str, Any]]:
        required_cat = required_cat if required_cat is not None else self.REQUIRED_CAT
        bbox_margin = bbox_margin if bbox_margin is not None else self.BBOX_MARGIN_PX

        with open(coco_json_path, "r") as f:
            coco = json.load(f)

        images = coco.get("images", [])
        annotations = coco.get("annotations", [])
        ann_by_image: Dict[Any, List[Dict[str, Any]]] = {}
        for ann in annotations:
            iid = ann.get("image_id")
            ann_by_image.setdefault(iid, []).append(ann)
        cats = coco.get("categories", [])
        catid_to_kpnames: Dict[int, List[str]] = {}
        for cat in cats:
            cid = cat.get("id")
            kpnames = cat.get("keypoints", []) or []
            catid_to_kpnames[cid] = kpnames

        trailing_num_re = re.compile(r'(\d+)$')

        def _canonical_group(name: str) -> str:
            n = re.sub(r'[^A-Za-z0-9_]', '_', name.lower()).strip('_')
            n = re.sub(r'(_|\-|\s)+\d+$', '', n)
            if 'jaw' in n:
                return 'jaw'
            if 'brow' in n or 'eyebrow' in n or 'eyebrows' in n:
                if 'left' in n or n.startswith('l_') or n.endswith('_l') or 'l' == n.split('_')[0]:
                    return 'left_eyebrow'
                if 'right' in n or n.startswith('r_') or n.endswith('_r') or 'r' == n.split('_')[0]:
                    return 'right_eyebrow'
                return 'eyebrows'
            if 'nose' in n:
                return 'nose'
            if 'eye' in n:
                if 'left' in n or n.startswith('l_') or n.endswith('_l') or 'l' == n.split('_')[0]:
                    return 'left_eye'
                if 'right' in n or n.startswith('r_') or n.endswith('_r') or 'r' == n.split('_')[0]:
                    return 'right_eye'
                return 'eyes'
            if 'mouth' in n or 'lip' in n:
                return 'mouth'
            return 'other'

        group_to_items: Dict[str, List[Tuple[int, str]]] = {}
        all_kpnames_seen: List[str] = []
        for cat in cats:
            kpnames = cat.get('keypoints', []) or []
            for kp in kpnames:
                all_kpnames_seen.append(kp)
                m = trailing_num_re.search(kp)
                if m:
                    num = int(m.group(1))
                    base = kp[:m.start()].rstrip('_- ')
                else:
                    num = None
                    base = kp
                grp = _canonical_group(base)
                group_to_items.setdefault(grp, []).append((num if num is not None else 99999, kp))

        desired_order = ['jaw', 'right_eyebrow', 'left_eyebrow', 'nose', 'right_eye', 'left_eye', 'mouth']

        ordered_kpnames: List[str] = []

        for g in desired_order:
            items = group_to_items.get(g, [])
            if items:
                items_sorted = sorted(items, key=lambda t: (t[0] if t[0] is not None else 99999, t[1]))
                ordered_kpnames.extend([name for _, name in items_sorted])

        def take_and_split(source_key: str, into_keys: Tuple[str, str], expected_counts: Tuple[int, int] = (5,5)):
            items = group_to_items.get(source_key, [])
            if not items:
                return
            items_sorted = sorted(items, key=lambda t: (t[0] if t[0] is not None else 99999, t[1]))
            names = [n for _, n in items_sorted]
            total = len(names)
            a, b = expected_counts
            if total == a + b:
                first = names[:a]; second = names[a:]
            else:
                mid = total // 2
                first = names[:mid]; second = names[mid:]
            ordered_kpnames.extend(first)
            ordered_kpnames.extend(second)

        if 'right_eyebrow' not in group_to_items and 'left_eyebrow' not in group_to_items:
            if 'eyebrows' in group_to_items:
                take_and_split('eyebrows', ('right_eyebrow', 'left_eyebrow'), expected_counts=(5,5))

        if 'right_eye' not in group_to_items and 'left_eye' not in group_to_items:
            if 'eyes' in group_to_items:
                take_and_split('eyes', ('right_eye', 'left_eye'), expected_counts=(6,6))

        if len(ordered_kpnames) < self.KEYPOINTS:
            for kp in all_kpnames_seen:
                if kp not in ordered_kpnames:
                    ordered_kpnames.append(kp)

        if len(ordered_kpnames) != self.KEYPOINTS:
            logger.warning("Assembled keypoint list length %d != KEYPOINTS (%d). Using concatenation fallback.",
                           len(ordered_kpnames), self.KEYPOINTS)
            ordered_kpnames = []
            for cat in sorted(cats, key=lambda c: c.get("id")):
                for kp in cat.get("keypoints", []) or []:
                    if kp not in ordered_kpnames:
                        ordered_kpnames.append(kp)
            if len(ordered_kpnames) < self.KEYPOINTS:
                while len(ordered_kpnames) < self.KEYPOINTS:
                    ordered_kpnames.append(f"kp_pad_{len(ordered_kpnames)}")
            else:
                ordered_kpnames = ordered_kpnames[:self.KEYPOINTS]

        kpname_to_index: Dict[str, int] = {}
        for idx, name in enumerate(ordered_kpnames):
            kpname_to_index[name] = idx

        logger.info("Built global keypoint ordering (%d): %s", len(ordered_kpnames),
                    ", ".join(ordered_kpnames[:min(20, len(ordered_kpnames))]) + ("..." if len(ordered_kpnames) > 20 else ""))

        out_entries: List[Dict[str, Any]] = []
        skipped_no_ann = 0
        skipped_few_kp = 0
        used = 0

        for img in tqdm(images, desc="Building cleaned annotations"):
            iid = img.get("id")
            anns = ann_by_image.get(iid, [])
            if not anns:
                skipped_no_ann += 1
                continue

            if not should_use_image_for_cat(anns, required_cat):
                continue

            global_flat = np.zeros((self.KEYPOINTS * 3,), dtype=np.float32)
            face_bbox = None
            for ann in anns:
                cat_id = ann.get("category_id")
                if ann.get("bbox") and (ann.get("num_keypoints", 0) == 0 or cat_id is None):
                    bbox = ann.get("bbox")
                    if bbox and len(bbox) >= 4:
                        bx, by, bw, bh = bbox[:4]
                        face_bbox = (int(bx), int(by), int(bx + bw), int(by + bh))
                flat = ann.get("keypoints", []) or []
                if not flat:
                    continue
                kpnames_local = catid_to_kpnames.get(cat_id, [])
                local_k = len(flat) // 3
                if kpnames_local and len(kpnames_local) == local_k:
                    for i in range(local_k):
                        name = kpnames_local[i]
                        if name in kpname_to_index:
                            gi = kpname_to_index[name]
                            global_flat[gi*3: gi*3+3] = flat[i*3:(i+1)*3]
                else:
                    for i in range(local_k):
                        zero_slots = np.where(global_flat.reshape(-1,3)[:,2] == 0)[0]
                        if len(zero_slots) == 0:
                            break
                        gi = zero_slots[0]
                        global_flat[gi*3: gi*3+3] = flat[i*3:(i+1)*3]

            vis = global_flat.reshape(-1,3)[:,2]
            visible_count = int((vis == 2).sum())
            if visible_count < max(1, int(0.01 * self.KEYPOINTS)):
                skipped_few_kp += 1
                continue

            if face_bbox is not None:
                xmin, ymin, xmax, ymax = face_bbox
            else:
                xs = global_flat.reshape(-1,3)[:,0]
                ys = global_flat.reshape(-1,3)[:,1]
                xs_vis = xs[vis > 0]
                ys_vis = ys[vis > 0]
                if xs_vis.size == 0 or ys_vis.size == 0:
                    skipped_few_kp += 1
                    continue
                xmin = int(np.floor(xs_vis.min())) - bbox_margin
                xmax = int(np.ceil(xs_vis.max())) + bbox_margin
                ymin = int(np.floor(ys_vis.min())) - bbox_margin
                ymax = int(np.ceil(ys_vis.max())) + bbox_margin

            W_img = img.get("width", None) or 0
            H_img = img.get("height", None) or 0
            xmin = max(0, xmin)
            ymin = max(0, ymin)
            xmax = min(W_img - 1 if W_img else xmax, int(xmax))
            ymax = min(H_img - 1 if H_img else ymax, int(ymax))
            if xmax <= xmin or ymax <= ymin:
                continue

            keypoints_pixels = []
            gf = global_flat.reshape(-1,3)
            for i_k in range(self.KEYPOINTS):
                xg, yg, vg = gf[i_k]
                keypoints_pixels.extend([float(xg), float(yg), float(vg)])

            raw_fname = img.get("file_name") or img.get("path") or ""
            pitch_key = None
            m = THERMAL_RE.search(os.path.basename(raw_fname))
            if m:
                pitch_key = str(abs(int(m.group("pitch"))))
            cls = int(self.pitch_to_emotion.get(pitch_key, 0)) if pitch_key is not None else 0

            entry = {
                "image_id": iid,
                "file_name": img.get("file_name"),
                "path": img.get("path"),
                "width": W_img,
                "height": H_img,
                "class": int(cls),
                "bbox": [int(xmin), int(ymin), int(xmax), int(ymax)],
                "keypoints": keypoints_pixels,
                "num_visible": visible_count
            }
            out_entries.append(entry)
            used += 1

        logger.info("Created annotations: used=%d skipped_no_ann=%d skipped_few_kp=%d",
                    used, skipped_no_ann, skipped_few_kp)
        out_dirname = os.path.dirname(out_json_path)
        if out_dirname:
            os.makedirs(out_dirname, exist_ok=True)
        with open(out_json_path, "w") as f:
            json.dump(out_entries, f, indent=2)
        logger.info("Wrote annotations JSON to %s", out_json_path)
        return out_entries

AnnotationsBuilderDefault = AnnotationsBuilder


def _ensure_path_exists_try_variants(images_root: str, entry: Dict[str, Any]) -> Optional[str]:
    fname = entry.get("file_name") or ""
    path_field = entry.get("path")
    candidates = []

    if path_field and os.path.isabs(path_field):
        candidates.append(path_field)

    if path_field:
        normalized = path_field.replace("\\", "/").lstrip("/\\")
        candidates.append(os.path.join(images_root, normalized))
        candidates.append(os.path.join(images_root, os.path.basename(normalized)))

    if fname:
        candidates.append(os.path.join(images_root, fname))
        candidates.append(os.path.join(images_root, os.path.basename(fname)))

    extracted_id = None
    try:
        extracted_id = int(entry.get("image_id"))
    except Exception:
        extracted_id = None
    if extracted_id is not None:
        candidates.append(os.path.join(images_root, str(extracted_id), fname))
        candidates.append(os.path.join(images_root, "images", str(extracted_id), fname))

    for c in candidates:
        if not c:
            continue
        c_norm = os.path.normpath(c)
        if os.path.exists(c_norm):
            return c_norm
    return None


def build_index_from_json(json_path: str,
                          images_root: str,
                          masks_root: Optional[str] = None,
                          keep_classes: Optional[Tuple[int, ...]] = (0, 1, 2, 3, 4, 5, 6, 7)
                          ) -> Tuple[List[Dict[str, Any]], int]:
    with open(json_path, "r") as f:
        entries = json.load(f)

    index = []
    keypoint_count = None
    for e in tqdm(entries, desc=f"Indexing {os.path.basename(json_path)}"):
        cls = int(e.get("class", 0))
        if keep_classes is not None and cls not in keep_classes:
            continue

        img_path = _ensure_path_exists_try_variants(images_root, e)
        if img_path is None:
            logger.warning("Could not resolve image path for entry: %s (skipping)", e.get("file_name"))
            continue
        found_mask = None
        if masks_root:
            base = os.path.splitext(os.path.basename(e.get("file_name") or ""))[0]
            cand_mask = os.path.join(masks_root, base + ".png")
            if os.path.exists(cand_mask):
                found_mask = os.path.normpath(cand_mask)
            else:
                extracted_id = str(e.get("image_id"))
                cand_mask2 = os.path.join(masks_root, extracted_id, base + ".png")
                if os.path.exists(cand_mask2):
                    found_mask = os.path.normpath(cand_mask2)

        kp = e.get("keypoints", [])
        if keypoint_count is None:
            if isinstance(kp, list) and len(kp) % 3 == 0 and len(kp) > 0:
                keypoint_count = len(kp) // 3
            else:
                raise RuntimeError("Cannot infer keypoint count from entry: " + str(e))

        index.append({
            "json_entry": e,
            "image_path": img_path,
            "mask_path": found_mask
        })

    if len(index) == 0:
        raise RuntimeError("No valid entries found in JSON after filtering classes and path resolution.")

    return index, int(keypoint_count)

def process_single_sample(entry_meta: Dict[str, Any],
                          keypoints_count: int,
                          out_size: Tuple[int, int] = (config.OUT_W, config.OUT_H),
                          normalize_class: bool = True,
                          num_classes: int = 8,
                          gauss_sigma: float = config.GAUSS_SIGMA,
                          normalize_landmarks: bool = False
                          ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.int32]:
    e = entry_meta["json_entry"]
    img_path = entry_meta["image_path"]
    mask_path = entry_meta.get("mask_path", None)

    im = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    if im is None:
        raise RuntimeError(f"cv2 failed to read {img_path}")
    if im.ndim == 3:
        im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    else:
        im_gray = im.copy()
    img = im_gray.astype(np.float32)
    if img.max() > 1.01:
        img = img / 255.0
    img = np.expand_dims(img, axis=-1)  # (H, W, 1)

    original_cls = int(e.get("class", 0))
    mapped_cls = original_cls
    if mapped_cls < 0 or mapped_cls >= num_classes:
        raise RuntimeError(f"Class mapping produced out-of-range label: {original_cls} -> {mapped_cls}")

    mask = None
    if mask_path:
        mask_full = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
        if mask_full is not None:
            if mask_full.ndim == 3:
                mask_full = cv2.cvtColor(mask_full, cv2.COLOR_BGR2GRAY)
            mask_arr = mask_full.astype(np.float32)
            if mask_arr.max() > 1.01:
                mask_arr = mask_arr / 255.0
            mask = np.expand_dims(mask_arr, axis=-1)

    if mask is None:
        mask = np.zeros_like(img, dtype=np.float32)

    kps = np.array(e.get('keypoints', [])).reshape(-1, 3)
    if kps.shape[0] != keypoints_count:
        if kps.size == 0:
            kps = np.zeros((keypoints_count, 3), dtype=float)
        else:
            kp_pad = np.zeros((keypoints_count, 3), dtype=float)
            kp_pad[:min(keypoints_count, kps.shape[0]), :] = kps[:min(keypoints_count, kps.shape[0]), :]
            kps = kp_pad

    ann_abs = []
    occl_flags = []
    H_img = float(img.shape[0])
    W_img = float(img.shape[1])
    for i_k in range(keypoints_count):
        xg, yg, vg = kps[i_k]
        x_px = float(np.clip(xg, 0.0, img.shape[1] - 1))
        y_px = float(np.clip(yg, 0.0, img.shape[0] - 1))
        if normalize_landmarks:
            x_val = x_px / (W_img + 1e-12)
            y_val = y_px / (H_img + 1e-12)
        else:
            x_val = x_px
            y_val = y_px
        ann_abs.extend([x_val, y_val])
        occl_flags.append(1.0 if int(vg) == 1 else 0.0)

    ann_flat = np.array(ann_abs, dtype=np.float32).reshape(-1)
    occl_arr = np.array(occl_flags, dtype=np.float32).reshape(-1)

    return img.astype(np.float32), mask.astype(np.float32), ann_flat.astype(np.float32), occl_arr.astype(np.float32), np.int32(mapped_cls)

def _py_generator(index: List[Dict[str, Any]],
                  keypoints_count: int,
                  out_size: Tuple[int, int],
                  normalize_class: bool,
                  num_classes: int,
                  gauss_sigma: float,
                  normalize_landmarks: bool = False,
                  attach_expression_weights: bool = False,
                  class_weights: Optional[np.ndarray] = None):
    for meta in index:
        try:
            img, mask, ann, occl, expr = process_single_sample(
                meta, keypoints_count,
                out_size=out_size,
                normalize_class=normalize_class,
                num_classes=num_classes,
                gauss_sigma=gauss_sigma,
                normalize_landmarks=normalize_landmarks
            )
            if attach_expression_weights:
                if class_weights is None:
                    raise RuntimeError("class_weights required when attach_expression_weights=True")
                cls = int(meta["json_entry"].get("class", 0))
                expr_w = float(class_weights[cls]) if cls < len(class_weights) else 0.0
                lw = 1.0
                sw = {'expression_output': float(expr_w), 'landmark_output': float(lw)}
                yield img, mask, ann, occl, expr, sw
            else:
                yield img, mask, ann, occl, expr
        except Exception as e:
            logger.warning("Skipping sample due to processing error: %s", e)
            continue

def build_tf_dataset(json_path: str,
                     images_root: str,
                     masks_root: Optional[str] = None,
                     out_size: Tuple[int, int] = [config.OUT_W, config.OUT_H],
                     batch_size: int = 16,
                     shuffle: bool = True,
                     shuffle_buffer: int = 512, 
                     repeat: bool = False,
                     normalize_class: bool = True,
                     num_classes: int = 8,
                     gauss_sigma: float = config.GAUSS_SIGMA,
                     normalize_landmarks: bool = False,
                     attach_expression_weights: bool = False,
                     keep_classes: Optional[Tuple[int, ...]] = None
                     ) -> tf.data.Dataset:
    index, keypoints_count = build_index_from_json(json_path, images_root, masks_root, keep_classes=keep_classes)
    if shuffle:
        random.shuffle(index)
    class_weights = None
    if attach_expression_weights:
        labels = np.array([int(e["json_entry"].get("class", 0)) for e in index], dtype=int)
        n = labels.shape[0]
        counts = np.bincount(labels, minlength=num_classes).astype(float)
        class_weights = np.zeros((num_classes,), dtype=float)
        for c in range(num_classes):
            if counts[c] > 0:
                class_weights[c] = float(n) / (float(num_classes) * counts[c])
            else:
                class_weights[c] = 0.0
        if num_classes > 0:
            class_weights[0] = 0.0

    gen = lambda: _py_generator(index, keypoints_count, out_size, normalize_class, num_classes, gauss_sigma,
                                normalize_landmarks,
                                attach_expression_weights=attach_expression_weights,
                                class_weights=class_weights)
    img_shape = (None, None, 1)
    mask_shape = (None, None, 1)
    ann_shape = (keypoints_count * 2,)
    occl_shape = (keypoints_count,)
    expr_shape = ()

    if attach_expression_weights:
        sample_weight_spec = {'expression_output': tf.TensorSpec(shape=(), dtype=tf.float32),
                              'landmark_output': tf.TensorSpec(shape=(), dtype=tf.float32)}
        output_signature = (
            tf.TensorSpec(shape=img_shape, dtype=tf.float32),
            tf.TensorSpec(shape=mask_shape, dtype=tf.float32),
            tf.TensorSpec(shape=ann_shape, dtype=tf.float32),
            tf.TensorSpec(shape=occl_shape, dtype=tf.float32),
            tf.TensorSpec(shape=expr_shape, dtype=tf.int32),
            sample_weight_spec
        )
    else:
        output_signature = (
            tf.TensorSpec(shape=img_shape, dtype=tf.float32),
            tf.TensorSpec(shape=mask_shape, dtype=tf.float32),
            tf.TensorSpec(shape=ann_shape, dtype=tf.float32),
            tf.TensorSpec(shape=occl_shape, dtype=tf.float32),
            tf.TensorSpec(shape=expr_shape, dtype=tf.int32),
        )

    ds = tf.data.Dataset.from_generator(gen, output_signature=output_signature)

    if repeat:
        ds = ds.repeat()

    ds = ds.batch(batch_size, drop_remainder=False)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds

def build_xy_dataset(json_path: str,
                     images_root: str,
                     masks_root: Optional[str] = None,
                     out_size: Tuple[int, int] = [config.OUT_W, config.OUT_H],
                     batch_size: int = 16,
                     shuffle: bool = True,
                     repeat: bool = False,
                     normalize_class: bool = True,
                     num_classes: int = 8,
                     gauss_sigma: float = config.GAUSS_SIGMA,
                     x_keys: Tuple[str, ...] = ("image",),
                     y_keys: Tuple[str, ...] = ("landmark", "occlusion"),
                     shuffle_buffer: int = 512,
                     normalize_landmarks: bool = False,
                     attach_expression_weights: bool = False,
                     keep_classes: Optional[Tuple[int, ...]] = None
                     ) -> tf.data.Dataset:
    if isinstance(x_keys, str):
        x_keys = (x_keys,)
    else:
        x_keys = tuple(x_keys)

    if isinstance(y_keys, str):
        y_keys = (y_keys,)
    else:
        y_keys = tuple(y_keys)

    x_keys = tuple(k.strip() for k in x_keys)
    y_keys = tuple(k.strip() for k in y_keys)

    valid_x = {"image", "mask"}
    valid_y = {"landmark", "occlusion", "mask", "expression"}
    for k in x_keys:
        if k not in valid_x:
            raise ValueError(f"Invalid x key: {k}. Valid: {valid_x}")
    for k in y_keys:
        if k not in valid_y:
            raise ValueError(f"Invalid y key: {k}. Valid: {valid_y}")

    ds = build_tf_dataset(json_path=json_path,
                          images_root=images_root,
                          masks_root=masks_root,
                          out_size=out_size,
                          batch_size=batch_size,
                          shuffle=shuffle,
                          shuffle_buffer=shuffle_buffer,
                          repeat=repeat,
                          normalize_class=normalize_class,
                          num_classes=num_classes,
                          gauss_sigma=gauss_sigma,
                          normalize_landmarks=normalize_landmarks,
                          attach_expression_weights=attach_expression_weights,
                          keep_classes=keep_classes)

    def _map_to_xy_tf_with_sw(image_batch, mask_batch, ann_batch, occl_batch, expr_batch, sw_batch):
        if len(x_keys) == 1:
            k = x_keys[0]
            if k == "image":
                x_out = tf.identity(image_batch)
            elif k == "mask":
                x_out = tf.identity(mask_batch)
            else:
                x_out = tf.identity(image_batch)
        else:
            out_dict = {}
            for k in x_keys:
                if k == "image":
                    out_dict["image"] = tf.identity(image_batch)
                elif k == "mask":
                    out_dict["mask"] = tf.identity(mask_batch)
            x_out = out_dict

        if len(y_keys) == 1:
            ky = y_keys[0]
            if ky == "landmark":
                y_out = tf.identity(ann_batch)
            elif ky == "occlusion":
                y_out = tf.identity(occl_batch)
            elif ky == "mask":
                y_out = tf.identity(mask_batch)
            elif ky in ("expression",):
                y_out = tf.identity(expr_batch)
            else:
                y_out = tf.identity(ann_batch)
        else:
            out_y = {}
            for ky in y_keys:
                if ky == "landmark":
                    out_y["landmark_output"] = tf.identity(ann_batch)
                elif ky == "occlusion":
                    out_y["occlusion_output"] = tf.identity(occl_batch)
                elif ky == "mask":
                    out_y["mask"] = tf.identity(mask_batch)
                elif ky in ("expression",):
                    out_y["expression_output"] = tf.identity(expr_batch)
            y_out = out_y

        return x_out, y_out, sw_batch

    def _map_to_xy_tf_no_sw(image_batch, mask_batch, ann_batch, occl_batch, expr_batch):
        if len(x_keys) == 1:
            k = x_keys[0]
            if k == "image":
                x_out = tf.identity(image_batch)
            elif k == "mask":
                x_out = tf.identity(mask_batch)
            else:
                x_out = tf.identity(image_batch)
        else:
            out_dict = {}
            for k in x_keys:
                if k == "image":
                    out_dict["image"] = tf.identity(image_batch)
                elif k == "mask":
                    out_dict["mask"] = tf.identity(mask_batch)
            x_out = out_dict

        if len(y_keys) == 1:
            ky = y_keys[0]
            if ky == "landmark":
                y_out = tf.identity(ann_batch)
            elif ky == "occlusion":
                y_out = tf.identity(occl_batch)
            elif ky == "mask":
                y_out = tf.identity(mask_batch)
            elif ky in ("expression",):
                y_out = tf.identity(expr_batch)
            else:
                y_out = tf.identity(ann_batch)
        else:
            out_y = {}
            for ky in y_keys:
                if ky == "landmark":
                    out_y["landmark_output"] = tf.identity(ann_batch)
                elif ky == "occlusion":
                    out_y["occlusion_output"] = tf.identity(occl_batch)
                elif ky == "mask":
                    out_y["mask"] = tf.identity(mask_batch)
                elif ky in ("expression",):
                    out_y["expression_output"] = tf.identity(expr_batch)
            y_out = out_y

        return x_out, y_out

    if attach_expression_weights:
        ds = ds.map(lambda a, b, c, d, e, sw: _map_to_xy_tf_with_sw(a, b, c, d, e, sw),
                    num_parallel_calls=tf.data.AUTOTUNE)
    else:
        ds = ds.map(lambda a, b, c, d, e: _map_to_xy_tf_no_sw(a, b, c, d, e),
                    num_parallel_calls=tf.data.AUTOTUNE)

    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds