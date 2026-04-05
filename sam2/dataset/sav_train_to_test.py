from __future__ import annotations
import argparse
from pathlib import Path
from typing import Dict, List, Optional
import json
import numpy as np
from PIL import Image
from tqdm import tqdm

# Optional imports (soft fail):
try:
    import cv2  # type: ignore
except Exception:
    cv2 = None  # type: ignore

try:
    from pycocotools import mask as mask_utils  # type: ignore
except Exception:
    mask_utils = None  # type: ignore

# -----------------------------
# Utils
# -----------------------------

def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def save_png_mask(path: Path, mask: np.ndarray) -> None:
    ensure_dir(path.parent)
    Image.fromarray((mask > 0).astype(np.uint8) * 255, mode="L").save(path)


def imwrite_jpg(path: Path, img_bgr: np.ndarray) -> None:
    ensure_dir(path.parent)
    ok = cv2.imwrite(str(path), img_bgr)
    if not ok:
        raise RuntimeError(f"Failed to write {path}")


def as_uint8_mask(arr: np.ndarray, H: int, W: int) -> np.ndarray:
    m = np.asarray(arr)
    if m.ndim > 2:
        axes = tuple(range(0, m.ndim - 2))
        m = m.max(axis=axes)
    if m.shape != (H, W):
        m = np.array(Image.fromarray((m > 0).astype(np.uint8) * 255, mode="L").resize((W, H), Image.NEAREST))
    if np.issubdtype(m.dtype, np.floating):
        m = (m > 0.5).astype(np.uint8)
    else:
        m = (m > 0).astype(np.uint8)
    return m

# -----------------------------
# Frame extraction (24fps default)
# -----------------------------

def extract_frames_24fps(video_path: Path, out_dir: Path, keep_native_fps: bool = False, target_fps: int = 24) -> tuple[int, int, int]:
    """Dump frames to JPG. Return (num_frames_written, H, W)."""
    if cv2 is None:
        raise ImportError("opencv-python is required for frame extraction")
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    native_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = 1.0
    if not keep_native_fps and native_fps > 0:
        step = max(1.0, round(native_fps / float(target_fps)))
    idx_native = 0
    idx_write = 0
    H = W = -1
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    pbar = tqdm(total=total, desc=f"frames {video_path.name}")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx_native % int(step) == 0:
            if H < 0:
                H, W = frame.shape[:2]
            out_path = out_dir / f"{idx_write:05d}.jpg"
            imwrite_jpg(out_path, frame)
            idx_write += 1
        idx_native += 1
        pbar.update(1)
    pbar.close()
    cap.release()
    if H < 0:
        raise RuntimeError(f"No frames extracted for {video_path}")
    return idx_write, H, W

# -----------------------------
# JSON parsing (FRAME‑MAJOR masklet)
# -----------------------------

def decode_rle_mask(rle_obj, H: int, W: int) -> np.ndarray:
    if mask_utils is None:
        raise ImportError("pycocotools is required to decode RLE")
    m = mask_utils.decode(rle_obj)
    if m.ndim == 3:
        m = np.any(m, axis=2)
    return as_uint8_mask(m, H, W)


def parse_masklet_frame_major(js: dict, H: int, W: int, Nframes24: int) -> Dict[int, List[np.ndarray]]:
    """Return dict: frame24 -> list of masks on that frame.
    Map t (0..T-1) uniformly to [0..N-1].
    """
    ml = js.get("masklet")
    if not isinstance(ml, list) or not ml:
        raise ValueError("JSON has no 'masklet' list or it's empty")
    T = len(ml)
    out: Dict[int, List[np.ndarray]] = {}
    for t, rles in enumerate(ml):
        if not isinstance(rles, list):
            continue
        if T == 1:
            f24 = 0
        else:
            f24 = int(round(t * (Nframes24 - 1) / (T - 1)))
        masks = []
        for rle in rles:
            if isinstance(rle, dict) and "counts" in rle:
                m = decode_rle_mask(rle, H, W)
                masks.append(m)
        if masks:
            out.setdefault(f24, []).extend(masks)
    return out

# -----------------------------
# Convert per sequence (INDEX‑BIND → INSTANCE‑MAJOR)
# -----------------------------

def convert_sequence(seq: str, src: Path, out_root: Path, prefer: str, fallback: Optional[str],
                     keep_native_fps: bool = False, target_fps: int = 24) -> None:
    # Locate mp4
    mp4 = None
    for p in [src / f"{seq}.mp4", src / "videos" / f"{seq}.mp4", src / seq / f"{seq}.mp4"]:
        if p.exists():
            mp4 = p; break
    if mp4 is None:
        raise FileNotFoundError(f"{seq}: cannot find MP4 under {src}")

    # Locate jsons (prefer + optional fallback)
    json_candidates: List[Path] = []
    def add_if_exists(p: Path):
        if p.exists():
            json_candidates.append(p)
    add_if_exists(src / f"{seq}_{prefer}.json")
    # Suggest not merge fallback first
    if fallback and fallback.lower() != "none":
        add_if_exists(src / f"{seq}_{fallback}.json")
    for sub in [src / "annotations", src / seq, src / "json"]:
        add_if_exists(sub / f"{seq}_{prefer}.json")
        if fallback and fallback.lower() != "none":
            add_if_exists(sub / f"{seq}_{fallback}.json")
    if not json_candidates:
        raise FileNotFoundError(f"{seq}: cannot find JSON ({prefer} / {fallback}) under {src}")

    # 1) Extract frames
    img_dir = out_root / "JPEGImages_24fps" / seq
    n_frames, H, W = extract_frames_24fps(mp4, img_dir, keep_native_fps=keep_native_fps, target_fps=target_fps)

    # 2) Read ONLY the first (prefer) JSON → frame_to_masks
    js = json.loads(json_candidates[0].read_text())
    frame_to_masks: Dict[int, List[np.ndarray]] = parse_masklet_frame_major(js, H, W, Nframes24=n_frames)

    # 3) Index‑bind to instance folders and save at 6fps grid
    ann_root = out_root / "Annotations_6fps" / seq
    # Evaluate maximum (K) of RLE for each frame
    Kmax = max((len(v) for v in frame_to_masks.values()), default=0)
    # Traverse each frame and write mask into j dirs of instance
    for f24 in sorted(frame_to_masks.keys()):
        f6 = (int(f24) // 4) * 4
        masks = frame_to_masks[f24]
        for j, m in enumerate(masks):
            inst_dir = ann_root / f"{j:03d}"
            save_png_mask(inst_dir / f"{f6:05d}.png", as_uint8_mask(m, H, W))

    # Print summary
    print(f"[SEQ {seq}] frames_with_labels={len(frame_to_masks)}  max_instances_per_frame={Kmax}")

# -----------------------------
# CLI
# -----------------------------

def parse_args():
    ap = argparse.ArgumentParser(description="SA‑V train (frame‑major masklet) → Test‑Pack instance‑major (index‑bind)")
    ap.add_argument("--src", required=True, type=Path, help="Root containing <seq>.mp4 and <seq>_{manual,auto}.json")
    ap.add_argument("--out", required=True, type=Path, help="Output root")
    ap.add_argument("--list", type=Path, help="Optional txt with sequence names; if omitted, enumerate mp4s")
    ap.add_argument("--prefer", choices=["manual", "auto"], default="manual", help="Prefer which annotations to use")
    ap.add_argument("--fallback", choices=["manual", "auto", "none"], default="none", help="Fallback; default none to avoid duplicates")
    ap.add_argument("--keep-native-fps", action="store_true", help="Keep native fps instead of resampling to 24fps")
    return ap.parse_args()


def discover_sequences(src: Path) -> List[str]:
    cands = []
    for p in list(src.glob("*.mp4")) + list((src / "videos").glob("*.mp4")) + list((src / "**").glob("*.mp4")):
        cands.append(p.stem)
    return sorted(set(cands))


def main():
    args = parse_args()
    out_root = args.out
    ensure_dir(out_root / "JPEGImages_24fps")
    ensure_dir(out_root / "Annotations_6fps")

    if args.list and args.list.exists():
        seqs = [x.strip() for x in args.list.read_text().splitlines() if x.strip()]
    else:
        seqs = discover_sequences(args.src)
        if not seqs:
            raise SystemExit("No <seq>.mp4 found; provide --list or check --src")

    for i, seq in enumerate(seqs, 1):
        print(f"[{i}/{len(seqs)}] {seq}")
        convert_sequence(
            seq=seq,
            src=args.src,
            out_root=out_root,
            prefer=args.prefer,
            fallback=(None if args.fallback == "none" else args.fallback),
            keep_native_fps=bool(args.keep_native_fps),
            target_fps=24,
        )

    (out_root / "sav_train.txt").write_text("\n".join(seqs) + "\n")
    print("\nDone. Wrote:")
    print(" - JPEGImages_24fps/<seq>/... .jpg")
    print(" - Annotations_6fps/<seq>/<inst>/<frame>.png (index‑bind)")
    print(" - sav_train.txt")


if __name__ == "__main__":
    main()
