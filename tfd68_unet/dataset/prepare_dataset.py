# tfd68_unet/prepare_dataset.py
import os
import json
import shutil
import logging
from typing import Optional, Dict, List, Any, Tuple
from tqdm import tqdm
import numpy as np
import cv2
import argparse

# local imports (try package import else local)
try:
    import config.config as config
    from dataset.datasetloader import AnnotationsBuilder, get_id_from_fname_safe, split_annotations, generate_masks
except Exception:
    import config.config as config
    from dataset.datasetloader import AnnotationsBuilder, get_id_from_fname_safe, split_annotations, generate_masks


logger = logging.getLogger("prepare_data")
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def _ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)
    return p


def copy_file_safely(src: str, dst: str):
    """Copy file, creating parent directories. If dst exists, skip copy."""
    _ensure_dir(os.path.dirname(dst))
    if os.path.exists(dst):
        return
    shutil.copy2(src, dst)


def _find_source_image(basename: str, images_root: str, try_ids: Optional[List[str]] = None) -> Optional[str]:
    """
    Try heuristics to find an image file for basename under images_root.
    - try_ids: optional list of id tokens to try as subfolders (e.g. ['1', '2'])
    """
    if not images_root:
        return None
    images_root = os.path.normpath(images_root)

    candidates = []
    candidates.append(os.path.join(images_root, basename))
    candidates.append(os.path.join(images_root, os.path.basename(basename)))

    if try_ids:
        for tid in try_ids:
            candidates.append(os.path.join(images_root, tid, basename))
            candidates.append(os.path.join(images_root, "images", tid, basename))
            candidates.append(os.path.join(images_root, str(tid), "images", basename))

    base_no_ext, _ = os.path.splitext(basename)
    for e in (".jpg", ".JPG", ".png", ".jpeg"):
        candidates.append(os.path.join(images_root, base_no_ext + e))
        if try_ids:
            for tid in try_ids:
                candidates.append(os.path.join(images_root, str(tid), base_no_ext + e))

    for cand in candidates:
        cand_norm = os.path.normpath(cand)
        if os.path.exists(cand_norm):
            return cand_norm

    # one-level subdir scan
    try:
        if os.path.isdir(images_root):
            for child in os.listdir(images_root):
                child_path = os.path.join(images_root, child)
                if not os.path.isdir(child_path):
                    continue
                cand = os.path.join(child_path, basename)
                if os.path.exists(cand):
                    return os.path.normpath(cand)
                for sub in ("images", "thermal", "visual"):
                    cand2 = os.path.join(child_path, sub, basename)
                    if os.path.exists(cand2):
                        return os.path.normpath(cand2)
    except Exception:
        pass

    # fallback recursive (slow)
    if os.path.isdir(images_root):
        for root, _, files in os.walk(images_root):
            if basename in files:
                return os.path.join(root, basename)

    return None


def prepare_data(
    coco_json: Optional[str] = None,
    out_json: Optional[str] = None,
    images_root: Optional[str] = None,
    masks_root: Optional[str] = None,
    target_dir: str = "data",
    splits: Optional[Dict[str, float]] = None,
    seed: Optional[int] = None,
    copy_masks: bool = False,
    remove_class_zero: bool = True,
    verbose: bool = True,
    resize_images: Optional[Tuple[int, int]] = None,
    generate_mask: bool = True,
    mask_mode: str = "disk",
    resize_masks_with_images: bool = True
) -> Dict[str, str]:
    """
    Prepare data layout for streaming loader.

    If resize_images is provided (W,H) then copied images are resized and the per-split JSON
    entries are updated so bbox/keypoints/width/height are consistent with the on-disk images.
    If copy_masks=True and resize_masks_with_images=True the masks will be resized to the same size.
    """
    coco_json = coco_json or getattr(config, "COCO_JSON", None)
    out_json = out_json or getattr(config, "OUT_JSON", "tfd68_annotations.json")
    images_root = images_root or getattr(config, "IMAGES_ROOT", "./input/images")
    masks_root = masks_root or getattr(config, "MASKS_ROOT", "./input/masks")
    splits = splits or getattr(config, "SPLIT", {"train": 0.6, "val": 0.2, "test": 0.2})
    seed = seed if seed is not None else getattr(config, "RANDOM_SEED", 42)

    # default resize to config values if resize_images is None
    if resize_images is None:
        resize_images = (getattr(config, "OUT_W", 256), getattr(config, "OUT_H", 256))

    _ensure_dir(target_dir)
    ab = AnnotationsBuilder()

    # 1) Create JSON if needed
    if not os.path.exists(out_json):
        if not coco_json or not os.path.exists(coco_json):
            raise FileNotFoundError("COCO JSON not found and JSON does not exist. Provide coco_json.")
        if verbose:
            logger.info("Creating JSON %s from COCO %s", out_json, coco_json)
        ab.create_standard_annotations(coco_json_path=coco_json, out_json_path=out_json,
                                       required_cat=getattr(config, "REQUIRED_CAT", None),
                                       bbox_margin=getattr(config, "BBOX_MARGIN_PX", None))
    else:
        if verbose:
            logger.info("Using existing JSON: %s", out_json)

    # 2) Split by id
    if verbose:
        logger.info("Splitting JSON by image id with splits=%s", splits)
    raw_splits = split_annotations(out_json, out_dir=target_dir, splits=splits, seed=seed, save_split_files=False, force_test_ids=[1,5,137])

    created_paths: Dict[str, str] = {}
    for split_name in ("train", "val", "test"):
        entries = raw_splits.get(split_name, [])
        if not entries:
            logger.warning("No entries for split %s (0 entries).", split_name)
            created_paths[split_name] = ""
            continue

        # Filter and prepare records
        filtered: List[Dict[str, Any]] = []
        for e in entries:
            orig_cls = int(e.get("class", 0))
            if remove_class_zero and orig_cls == 0:
                continue

            orig_fname = e.get("file_name") or e.get("path") or ""
            img_id_token = get_id_from_fname_safe(os.path.basename(orig_fname))
            if img_id_token is not None:
                try:
                    subject_id = int(img_id_token)
                except Exception:
                    subject_id = int(e.get("image_id", 0))
            else:
                subject_id = int(e.get("image_id", 0))

            basename = os.path.basename(orig_fname) if orig_fname else f"{e.get('image_id')}.jpg"
            rec = dict(e)  # shallow copy
            rec["subject_id"] = int(subject_id)
            rec["file_name"] = basename
            rec["path"] = os.path.join(split_name, "images", basename)
            # we'll possibly update bbox/keypoints/width/height later during copy
            filtered.append(rec)

        # Make output dirs
        split_dir = os.path.join(target_dir, split_name)
        images_out_dir = os.path.join(split_dir, "images")
        masks_out_dir = os.path.join(split_dir, "masks") if copy_masks else None
        os.makedirs(images_out_dir, exist_ok=True)
        if copy_masks:
            os.makedirs(masks_out_dir, exist_ok=True)

        # Copy images and optionally resize; update rec entries appropriately
        copied = 0
        missing = 0
        updated_filtered: List[Dict[str, Any]] = []
        for rec in tqdm(filtered, desc=f"Copy {split_name} images"):
            basename = rec.get("file_name")
            # find original entry corresponding to this rec (by basename or image_id)
            original_entry = None
            for e in entries:
                if os.path.basename(e.get("file_name") or e.get("path") or "") == basename or int(e.get("image_id", -1)) == int(rec.get("image_id", -2)):
                    original_entry = e
                    break

            found = None
            if original_entry:
                raw_path_field = original_entry.get("path")
                raw_file_field = original_entry.get("file_name")
                if raw_path_field and os.path.isabs(raw_path_field) and os.path.exists(raw_path_field):
                    found = raw_path_field
                else:
                    if raw_path_field:
                        cand = os.path.normpath(os.path.join(images_root, raw_path_field.lstrip("/\\")))
                        if os.path.exists(cand):
                            found = cand
                    if found is None and raw_file_field:
                        cand = os.path.normpath(os.path.join(images_root, raw_file_field))
                        if os.path.exists(cand):
                            found = cand

            if found is None:
                try_ids = [str(rec.get("subject_id"))] if rec.get("subject_id") is not None else None
                found = _find_source_image(basename, images_root, try_ids=try_ids)

            if found is None:
                logger.warning("Could not find source image for %s (basename=%s). Skipping.", rec.get("file_name"), basename)
                missing += 1
                continue

            dst_path = os.path.join(images_out_dir, basename)
            try:
                # If resize_images is requested - load and resize and save; otherwise copy file
                if resize_images:
                    new_w, new_h = resize_images
                    im = cv2.imread(found, cv2.IMREAD_UNCHANGED)
                    if im is None:
                        raise RuntimeError(f"Failed to read {found}")

                    # source image size
                    src_h, src_w = im.shape[:2]

                    # Default orig bbox = full image (will be overwritten if bbox exists)
                    bx0, by0, bx1, by1 = 0, 0, src_w - 1, src_h - 1

                    # try to get bbox from original_entry if available
                    raw_bbox = None
                    if original_entry:
                        raw_bbox = original_entry.get("bbox", None) or original_entry.get("bbox_xyxy", None)

                    if raw_bbox and len(raw_bbox) >= 4:
                        # raw_bbox may be [x,y,w,h] or [x0,y0,x1,y1]
                        bx0_raw, by0_raw, b2, b3 = raw_bbox[0], raw_bbox[1], raw_bbox[2], raw_bbox[3]
                        # detect format: treat as (x,w) if third entry is width (i.e. small) else as x1
                        try:
                            # Heuristic detection similar to original code:
                            if (b2 <= bx0_raw) or (b2 > max(src_w, src_h) * 2):
                                # likely [x,y,w,h]
                                bw, bh = b2, b3
                                bx1_raw = bx0_raw + bw
                                by1_raw = by0_raw + bh
                            else:
                                # likely [x0,y0,x1,y1]
                                bx1_raw = b2
                                by1_raw = b3
                            # round/clamp to integer pixel coords
                            bx0 = int(round(bx0_raw))
                            by0 = int(round(by0_raw))
                            bx1 = int(round(bx1_raw))
                            by1 = int(round(by1_raw))
                        except Exception:
                            bx0, by0, bx1, by1 = 0, 0, src_w - 1, src_h - 1

                    # clamp bbox to source image
                    bx0 = max(0, min(bx0, src_w - 1))
                    by0 = max(0, min(by0, src_h - 1))
                    bx1 = max(0, min(bx1, src_w - 1))
                    by1 = max(0, min(by1, src_h - 1))

                    # ensure valid crop, else fallback to full image
                    if bx1 <= bx0 + 1 or by1 <= by0 + 1:
                        bx0, by0, bx1, by1 = 0, 0, src_w - 1, src_h - 1

                    crop_w = (bx1 - bx0) + 1
                    crop_h = (by1 - by0) + 1

                    # Crop from source image then resize the crop to target (new_w, new_h)
                    crop = im[by0:by1 + 1, bx0:bx1 + 1].copy()
                    if crop.size == 0:
                        # fallback to resizing full image
                        crop = im.copy()
                        crop_h, crop_w = crop.shape[:2]
                        bx0, by0, bx1, by1 = 0, 0, src_w - 1, src_h - 1

                    im_out = cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_AREA)
                    cv2.imwrite(dst_path, im_out)

                    # Update metadata for this record: width/height are those of the saved (cropped/resized) image
                    rec["width"] = int(new_w)
                    rec["height"] = int(new_h)

                    # Save mapping back to original: allow denormalization later
                    rec["orig_image_path"] = os.path.normpath(found)
                    rec["orig_bbox"] = [int(bx0), int(by0), int(bx1), int(by1)]
                    rec["orig_size"] = [int(src_w), int(src_h)]

                    # scaling factors from crop space -> saved resized image (for converting absolute px)
                    scale_crop_x = float(new_w) / float(crop_w) if crop_w > 0 else 1.0
                    scale_crop_y = float(new_h) / float(crop_h) if crop_h > 0 else 1.0

                    if verbose:
                        logger.info("Cropped '%s' bbox %s from source (W,H)=(%d,%d) -> crop (W,H)=(%d,%d) -> resized to (W,H)=(%d,%d)",
                                    basename, rec["orig_bbox"], src_w, src_h, crop_w, crop_h, new_w, new_h)

                    # update keypoints if present: convert original keypoints (absolute source px) to
                    # coordinates in the saved crop/resized image (absolute px), preserving visibility if present.
                    if original_entry:
                        kpf = original_entry.get("keypoints", []) or []
                        if kpf:
                            arr = np.array(kpf, dtype=float).reshape(-1, 3)
                            # arr[:, 0] = x, arr[:, 1] = y, arr[:, 2] = v (COCO style)
                            # Shift to crop coordinates then scale to new_w/new_h
                            arr[:, 0] = (arr[:, 0] - bx0) * scale_crop_x
                            arr[:, 1] = (arr[:, 1] - by0) * scale_crop_y
                            # clamp into [0, new_w-1] / [0, new_h-1]
                            arr[:, 0] = np.clip(arr[:, 0], 0, new_w - 1)
                            arr[:, 1] = np.clip(arr[:, 1], 0, new_h - 1)
                            rec["keypoints"] = [float(x) for x in arr.reshape(-1).tolist()]
                            if verbose:
                                logger.info("Updated %d keypoints to cropped/resized image coords (first kp after transform: %s)",
                                            arr.shape[0], arr[0, :].tolist())
                        else:
                            # no keypoints -> keep empty
                            rec["keypoints"] = []
                    else:
                        rec["keypoints"] = []

                else:
                    # no resize requested -> just copy original file and keep metadata consistent
                    copy_file_safely(found, dst_path)
                    im = cv2.imread(dst_path, cv2.IMREAD_UNCHANGED)
                    if im is not None:
                        h, w = im.shape[:2]
                        rec["width"] = int(w)
                        rec["height"] = int(h)
                    else:
                        # fallback to original metadata if available
                        if original_entry:
                            rec["width"] = int(original_entry.get("width", 0) or 0)
                            rec["height"] = int(original_entry.get("height", 0) or 0)

                copied += 1
            except Exception as ex:
                logger.warning("Failed to copy/resize %s -> %s: %s", found, dst_path, ex)
                missing += 1
                continue

            # copy mask if requested
            if copy_masks and masks_root:
                mask_basename_png = os.path.splitext(basename)[0] + ".png"
                mask_found = None
                possible = [
                    os.path.join(masks_root, mask_basename_png),
                    os.path.join(masks_root, basename.replace(".jpg", ".png")),
                    os.path.join(masks_root, basename.replace(".JPG", ".png")),
                    os.path.join(masks_root, str(rec.get("subject_id")), mask_basename_png),
                    os.path.join(masks_root, "images", str(rec.get("subject_id")), mask_basename_png)
                ]
                for pm in possible:
                    if os.path.exists(pm):
                        mask_found = pm
                        break
                if mask_found:
                    try:
                        dst_mask_path = os.path.join(masks_out_dir, mask_basename_png)
                        mask_im = cv2.imread(mask_found, cv2.IMREAD_UNCHANGED)
                        if mask_im is None:
                            # fallback to copy
                            copy_file_safely(mask_found, dst_mask_path)
                        else:
                            # crop to same bbox and resize to new_w,new_h (use nearest to preserve mask labels)
                            mh, mw = mask_im.shape[:2]
                            # Use same bx0,by0,bx1,by1 as computed for the image. If bbox not present, will be full image.
                            # Clamp bbox to mask dims
                            mbx0 = max(0, min(bx0, mw - 1))
                            mby0 = max(0, min(by0, mh - 1))
                            mbx1 = max(0, min(bx1, mw - 1))
                            mby1 = max(0, min(by1, mh - 1))
                            mask_crop = mask_im[mby0:mby1 + 1, mbx0:mbx1 + 1].copy()
                            if mask_crop.size == 0:
                                # fallback to resize full mask
                                mask_out = cv2.resize(mask_im, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
                            else:
                                mask_out = cv2.resize(mask_crop, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
                            cv2.imwrite(dst_mask_path, mask_out)
                            if verbose:
                                logger.info("Mask %s cropped/resized -> %s", mask_found, dst_mask_path)
                    except Exception as ex:
                        logger.warning("Failed to copy mask %s -> %s: %s", mask_found, dst_mask_path, ex)

            updated_filtered.append(rec)

        logger.info("Split %s: copied=%d missing=%d final_count=%d", split_name, copied, missing, len(updated_filtered))

        # Save filtered JSON for this split into target_dir (without helper fields)
        out_json_split = os.path.join(target_dir, f"{split_name}.json")
        with open(out_json_split, "w") as f:
            json.dump(updated_filtered, f, indent=2)
        logger.info("Wrote %d entries to %s", len(updated_filtered), out_json_split)

        # Log a short example to verify annotation consistency
        if len(updated_filtered) > 0:
            ex = updated_filtered[0]
            logger.info("Sample saved entry (split=%s): file=%s width=%s height=%s bbox=%s num_kps=%d",
                        split_name, ex.get("file_name"), ex.get("width"), ex.get("height"), ex.get("bbox"), len(ex.get("keypoints") or [])//3)

        created_paths[split_name] = out_json_split

        if generate_mask:
            try:
                mask_radius = config.MASK_RADIUS
                splits_to_gen = tuple([s for s, p in created_paths.items() if p and os.path.exists(p)])
                logger.info(f"Auto-generating masks in {target_dir} for splits={splits_to_gen} (mode={mask_mode}, radius={mask_radius})")
                generate_masks(
                    prepared_dir=target_dir,
                    splits=splits_to_gen,
                    out_size=resize_images,
                    mode=mask_mode,
                    radius=mask_radius,
                    overwrite=True,
                    verbose=verbose
                )
                logger.info("Mask generation completed successfully.")
            except Exception as ex:
                logger.warning(f"Mask generation failed: {ex}")

    return created_paths


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--coco", type=str, default=getattr(config, "COCO_JSON", None))
    parser.add_argument("--out_json", type=str, default=getattr(config, "OUT_JSON", "tfd68_annotations.json"))
    parser.add_argument("--images_root", type=str, default=getattr(config, "IMAGES_ROOT", "./input/images"))
    parser.add_argument("--masks_root", type=str, default=getattr(config, "MASKS_ROOT", "./input/masks"))
    parser.add_argument("--target_dir", type=str, default=getattr(config, "PREPARED_DATA_DIR", "data"))
    parser.add_argument("--copy_masks", action="store_true")
    parser.add_argument("--remove_class_zero", action="store_true")
    parser.add_argument("--resize", nargs=2, type=int, metavar=("W", "H"), help="If provided, resize images to W H when copying and update annotations.")
    parser.add_argument("--generate_mask", type=bool, default=True)
    args = parser.parse_args()

    # respect CLI override; otherwise use config.OUT_W / config.OUT_H
    resize_images = tuple(args.resize) if args.resize else (getattr(config, "OUT_W", 256), getattr(config, "OUT_H", 256))

    created = prepare_data(
        coco_json=args.coco,
        out_json=args.out_json,
        images_root=args.images_root,
        masks_root=args.masks_root,
        target_dir=args.target_dir,
        copy_masks=args.copy_masks,
        remove_class_zero=args.remove_class_zero,
        resize_images=resize_images,
        generate_mask=args.generate_mask,
        resize_masks_with_images=True
    )
    print("Prepared JSONs:", created)


if __name__ == "__main__":
    main()

    