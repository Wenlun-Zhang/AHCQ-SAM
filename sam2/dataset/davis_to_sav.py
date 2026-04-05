#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path
from typing import Dict, List, Set

import numpy as np
from PIL import Image
from tqdm import tqdm


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def link_or_copy(src: Path, dst: Path, mode: str = "symlink") -> None:
    ensure_dir(dst.parent)
    if dst.exists():
        return
    if mode == "symlink":
        try:
            rel = os.path.relpath(src, start=dst.parent)
            dst.symlink_to(rel)
        except Exception:
            dst.symlink_to(src)
    elif mode == "copy":
        shutil.copy2(src, dst)
    else:
        raise ValueError(f"Unknown image mode: {mode}")


def read_seq_list(davis_root: Path, year: str, split: str) -> List[str]:
    txt = davis_root / "ImageSets" / year / f"{split}.txt"
    if not txt.exists():
        raise FileNotFoundError(f"Split file not found: {txt}")
    return [l.strip() for l in txt.read_text().splitlines() if l.strip()]


def load_mask(path: Path) -> np.ndarray:
    m = np.array(Image.open(path))
    if m.ndim == 3:
        m = m[..., 0]
    return m


def save_bin_mask(path: Path, bin_mask: np.ndarray) -> None:
    ensure_dir(path.parent)
    Image.fromarray((bin_mask > 0).astype(np.uint8) * 255, mode="L").save(path)


def collect_object_ids(mask_paths: List[Path]) -> List[int]:
    ids: Set[int] = set()
    for mp in mask_paths:
        m = load_mask(mp)
        for v in np.unique(m).tolist():
            v = int(v)
            if v != 0:
                ids.add(v)
    return sorted(ids)


def davis_split_to_pack(
    davis_root: Path,
    out_pack_root: Path,   # e.g. out_root / sav_train
    year: str,
    split: str,
    resolution: str = "480p",
    image_mode: str = "symlink",
    max_seqs: int | None = None,
) -> None:
    """
    Convert ONE split (train or val) into SA-V-like pack under out_pack_root.
      out_pack_root/
        JPEGImages_24fps/<seq>/*.jpg
        Annotations_6fps/<seq>/<inst_id>/*.png
        <out_pack_root.name>.txt
    """
    seqs = read_seq_list(davis_root, year, split)
    if max_seqs is not None:
        seqs = seqs[:max_seqs]

    jpeg_src_root = davis_root / "JPEGImages" / resolution
    ann_src_root = davis_root / "Annotations" / resolution

    jpeg_dst_root = out_pack_root / "JPEGImages_24fps"
    ann_dst_root = out_pack_root / "Annotations_6fps"
    ensure_dir(jpeg_dst_root)
    ensure_dir(ann_dst_root)

    print(f"\n[DAVIS->PACK] split={split}  seqs={len(seqs)}  out={out_pack_root}")

    for seq in seqs:
        seq_img_dir = jpeg_src_root / seq
        seq_ann_dir = ann_src_root / seq
        if not seq_img_dir.exists():
            raise FileNotFoundError(f"Missing JPEGImages dir: {seq_img_dir}")
        if not seq_ann_dir.exists():
            raise FileNotFoundError(f"Missing Annotations dir: {seq_ann_dir}")

        frame_paths = sorted(seq_img_dir.glob("*.jpg"))
        if not frame_paths:
            frame_paths = sorted(seq_img_dir.glob("*.png"))
        if not frame_paths:
            raise RuntimeError(f"No frames found in: {seq_img_dir}")

        mask_paths = sorted(seq_ann_dir.glob("*.png"))
        if not mask_paths:
            raise RuntimeError(f"No masks found in: {seq_ann_dir}")

        # Determine object ids
        if year == "2016":
            obj_ids = [1]
            id_to_inst = {1: "000"}
        else:
            obj_ids = collect_object_ids(mask_paths)
            if not obj_ids:
                obj_ids = [1]
            id_to_inst = {oid: f"{i:03d}" for i, oid in enumerate(obj_ids)}

        # Copy/link images
        dst_img_dir = jpeg_dst_root / seq
        ensure_dir(dst_img_dir)
        for fp in frame_paths:
            link_or_copy(fp, dst_img_dir / fp.name, mode=image_mode)

        # Per-instance masks
        dst_ann_seq = ann_dst_root / seq
        ensure_dir(dst_ann_seq)

        mask_map: Dict[str, Path] = {mp.stem: mp for mp in mask_paths}
        stems = sorted(mask_map.keys(), key=lambda s: int(s) if s.isdigit() else s)

        pbar = tqdm(stems, desc=f"[{split}] {seq}", leave=False)
        for stem in pbar:
            m = load_mask(mask_map[stem])

            if year == "2016":
                bin_mask = (m > 0).astype(np.uint8)
                save_bin_mask(dst_ann_seq / "000" / f"{stem}.png", bin_mask)
            else:
                for oid in obj_ids:
                    bin_mask = (m == oid).astype(np.uint8)
                    if bin_mask.any():
                        save_bin_mask(dst_ann_seq / id_to_inst[oid] / f"{stem}.png", bin_mask)

    # Write list file compatible with your main code:
    # train_suffix = args.sav_train.name -> expects "<pack_name>.txt"
    out_list = out_pack_root / f"{out_pack_root.name}.txt"
    out_list.write_text("\n".join(seqs) + "\n")
    print(f"[DAVIS->PACK] wrote list: {out_list}")


def main():
    ap = argparse.ArgumentParser("Convert DAVIS train+val to SA-V-like pack structure")
    ap.add_argument("--davis-root", type=Path, required=True, help="DAVIS root (contains JPEGImages/Annotations/ImageSets)")
    ap.add_argument("--out-root", type=Path, required=True, help="Output root. Will create subfolders: sav_train/ sav_val/")
    ap.add_argument("--year", type=str, default="2017", choices=["2016", "2017"])
    ap.add_argument("--resolution", type=str, default="480p")
    ap.add_argument("--image-mode", type=str, default="symlink", choices=["symlink", "copy"])
    ap.add_argument("--max-seqs", type=int, default=None, help="Debug: only first N sequences per split")
    args = ap.parse_args()

    ensure_dir(args.out_root)

    # create two packs under one root (matches your main code expectation nicely)
    train_pack = args.out_root / "sav_train"
    val_pack = args.out_root / "sav_val"
    ensure_dir(train_pack)
    ensure_dir(val_pack)

    davis_split_to_pack(
        davis_root=args.davis_root,
        out_pack_root=train_pack,
        year=args.year,
        split="train",
        resolution=args.resolution,
        image_mode=args.image_mode,
        max_seqs=args.max_seqs,
    )
    davis_split_to_pack(
        davis_root=args.davis_root,
        out_pack_root=val_pack,
        year=args.year,
        split="val",
        resolution=args.resolution,
        image_mode=args.image_mode,
        max_seqs=args.max_seqs,
    )

    print("\n[DAVIS->PACK] Done.")
    print(f"  Train pack: {train_pack}")
    print(f"  Val pack:   {val_pack}")
    print("  Each pack contains:")
    print("    - JPEGImages_24fps/<seq>/*.jpg")
    print("    - Annotations_6fps/<seq>/<inst_id>/*.png")
    print("    - <pack_name>.txt")


if __name__ == "__main__":
    main()
