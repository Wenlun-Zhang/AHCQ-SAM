from __future__ import annotations

import argparse
from copy import deepcopy
import json
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image
import torch
from scipy.ndimage import binary_dilation, binary_erosion
from tqdm import tqdm

BASELINE_ROOT = Path(__file__).resolve().parent
BASELINES_ROOT = BASELINE_ROOT.parent
REPO_ROOT = BASELINES_ROOT.parent
ROOT = REPO_ROOT / "sam2"
SAM2_REPO = ROOT / "sam2"
AHCQSAM_ROOT = ROOT / "ahcqsam"
for path in (BASELINES_ROOT, ROOT, SAM2_REPO, AHCQSAM_ROOT, BASELINE_ROOT):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from sam2.build_sam import build_sam2_video_predictor  # noqa: E402

from model.quant_model import bimodal_adjust, specials  # noqa: E402
from quantization.observer import ObserverBase  # noqa: E402
from quantization.quantized_module import (  # noqa: E402
    PreQuantizedLayer,
    QuantizedBlock,
    QuantizedLayer,
    QuantizedMatMul,
)
from quantization.state import (  # noqa: E402
    disable_all,
    enable_calibration_woquantization,
    enable_quantization,
)
from solver import utils as quant_utils  # noqa: E402
from solver.recon import reconstruction  # noqa: E402

from compute_allocation import (  # noqa: E402
    apply_mixed_precision_bits,
    search_mixed_precision_bits,
)
from sis import get_sis_stats, maybe_apply_sis, reset_sis_stats  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Mix-QSAM2-style PTQ for SAM2 video segmentation on SA-V.")
    parser.add_argument("--sav-train", type=Path, default=Path("/home/dataset/SA-V/sav_train"))
    parser.add_argument("--sav-test", type=Path, default=Path("/home/dataset/SA-V/sav_test"))
    parser.add_argument("--model-cfg", type=str, default="configs/sam2.1/sam2.1_hiera_t.yaml")
    parser.add_argument("--checkpoint", type=Path, default=SAM2_REPO / "checkpoints" / "sam2.1_hiera_tiny.pt")
    parser.add_argument("--q_config", type=str, default="./mix_qsam2/exp_video/config44.yaml")
    parser.add_argument("--fp-model", action="store_true", default=False)
    parser.add_argument("--recon", action="store_true", default=False)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--calib-seed", type=int, default=0)
    parser.add_argument("--pred-root", type=Path, default=None)
    parser.add_argument("--save-json", type=Path, default=None)
    parser.add_argument("--save-bit-allocation", type=Path, default=None)
    return parser.parse_args()


def read_list(txt: Path) -> List[str]:
    return [x.strip() for x in txt.read_text().splitlines() if x.strip()]


def save_mask(path: Path, mask: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((mask > 0).astype(np.uint8) * 255, mode="L").save(path)


def boundary(mask: np.ndarray, dilation_ratio: float = 0.02) -> np.ndarray:
    h, w = mask.shape
    diag = float(np.hypot(h, w))
    iters = max(1, int(round(dilation_ratio * diag)))
    mask = mask > 0
    return np.logical_xor(
        binary_dilation(mask, iterations=iters),
        binary_erosion(mask, iterations=iters),
    )


def iou(pred: np.ndarray, gt: np.ndarray | None) -> float:
    if gt is None:
        return np.nan
    pred = pred > 0
    gt = gt > 0
    union = np.logical_or(pred, gt).sum()
    inter = np.logical_and(pred, gt).sum()
    return (inter / union) if union > 0 else (1.0 if inter == 0 else 0.0)


def f_measure(pred: np.ndarray, gt: np.ndarray | None, eps: float = 1e-6) -> float:
    if gt is None:
        return np.nan
    pred_boundary = boundary(pred)
    gt_boundary = boundary(gt)
    inter = np.logical_and(pred_boundary, gt_boundary).sum()
    precision = (inter + eps) / (pred_boundary.sum() + eps)
    recall = (inter + eps) / (gt_boundary.sum() + eps)
    return (2 * precision * recall) / (precision + recall) if precision + recall > 0 else 0.0


class SAVJPEGIndex:
    def __init__(self, sample: str, sav_root: Path):
        self.sample = sample
        self.sav_root = sav_root
        self.img_dir = sav_root / "JPEGImages_24fps" / sample
        self.ann_dir = sav_root / "Annotations_6fps" / sample
        if not self.img_dir.exists():
            raise FileNotFoundError(f"Frames folder missing: {self.img_dir}")
        if not self.ann_dir.exists():
            raise FileNotFoundError(f"Annotations folder missing: {self.ann_dir}")

        self.frame_paths = sorted(self.img_dir.glob("*.jpg"))
        if not self.frame_paths:
            if list(self.img_dir.glob("*.png")):
                raise RuntimeError(f"{self.sample}: PNG frames detected; convert to JPG first.")
            raise RuntimeError(f"{self.sample}: no JPG frames found under {self.img_dir}")

        self.inst_to_frames: Dict[str, List[int]] = {}
        for subdir in sorted(path for path in self.ann_dir.iterdir() if path.is_dir()):
            frame_ids: List[int] = []
            for mask_file in sorted(subdir.glob("*.png")):
                try:
                    frame_ids.append(int(mask_file.stem))
                except ValueError:
                    continue
            if frame_ids:
                self.inst_to_frames[subdir.name] = sorted(set(frame_ids))

        if not self.inst_to_frames:
            raise RuntimeError(f"{self.sample}: no annotated instances found under {self.ann_dir}")

        self.H, self.W = np.array(Image.open(self.frame_paths[0]).convert("L")).shape

    def first_prompt(self) -> Tuple[int, str, np.ndarray]:
        best: Tuple[int, str, np.ndarray] | None = None
        for inst_id, frame_ids in self.inst_to_frames.items():
            if not frame_ids:
                continue
            frame_id = frame_ids[0]
            mask = load_mask(self.ann_dir / inst_id / f"{frame_id:05d}.png")
            if best is None or frame_id < best[0]:
                best = (frame_id, inst_id, mask)
        if best is None:
            raise RuntimeError(f"{self.sample}: could not find any annotated frame to prompt.")
        return best


class SAVVideoDataset:
    def __init__(self, sav_root: Path, list_file: Path):
        self.sav_root = sav_root
        self.sample_names = read_list(list_file)

    def __len__(self) -> int:
        return len(self.sample_names)

    def __getitem__(self, idx: int) -> SAVJPEGIndex:
        return SAVJPEGIndex(self.sample_names[idx], self.sav_root)


def load_mask(path: Path) -> np.ndarray:
    mask = np.array(Image.open(path))
    if mask.ndim == 3:
        mask = mask[..., 0]
    return (mask > 0).astype(np.uint8)


def build_dataset(sav_root: Path) -> SAVVideoDataset:
    list_file = sav_root / f"{sav_root.name}.txt"
    return SAVVideoDataset(sav_root, list_file)


def build_predictor(args: argparse.Namespace):
    predictor = build_sam2_video_predictor(
        args.model_cfg,
        str(args.checkpoint),
        device=args.device,
    )
    predictor.eval()
    return predictor


def neutral_ahcqsam_config():
    return quant_utils.EasyDict(
        {
            "acnr": False,
            "k": 0,
            "cag": False,
            "group": 1,
            "hluq": False,
            "lnq": False,
        }
    )


def normalize_quant_config(q_config):
    if "ahcqsam" in q_config:
        del q_config["ahcqsam"]
    q_config.ahcqsam = neutral_ahcqsam_config()
    return q_config


def quantize_model(model: torch.nn.Module, q_config) -> torch.nn.Module:
    def replace_module(module):
        for name, child in module.named_children():
            if type(child) in specials:
                setattr(
                    module,
                    name,
                    specials[type(child)](
                        child,
                        q_config.w_qconfig,
                        q_config.a_qconfig,
                        q_config.ahcqsam,
                        q_config.ptq4sam,
                    ),
                )
            else:
                replace_module(child)

    replace_module(model)
    return model


def set_observer_names(model: torch.nn.Module) -> None:
    for name, module in model.named_modules():
        if isinstance(module, ObserverBase):
            module.set_name(name)


def print_quant_settings(q_config) -> None:
    print("=== Quantization Settings ===")
    print(f"a_qconfig.bit: {q_config.a_qconfig.bit}")
    print(f"w_qconfig.bit: {q_config.w_qconfig.bit}")
    print(f"calibrate.sample: {q_config.calibrate.sample}")
    print(f"calibrate.frame: {q_config.calibrate.frame}")
    print(f"ptq4sam.BIG: {q_config.ptq4sam.BIG}")
    print(f"ptq4sam.AGQ: {q_config.ptq4sam.AGQ}")
    if hasattr(q_config, "mix_qsam2"):
        print(f"mix_qsam2.enabled: {q_config.mix_qsam2.enabled}")


def collect_prompts(index: SAVJPEGIndex) -> Tuple[int, np.ndarray, List[Tuple[int, int, np.ndarray]], Dict[str, int]]:
    inst_ids = sorted(index.inst_to_frames.keys())
    inst_to_oid = {inst_id: i + 1 for i, inst_id in enumerate(inst_ids)}
    t0, inst0, mask0 = index.first_prompt()

    prompts: List[Tuple[int, int, np.ndarray]] = [(inst_to_oid[inst0], t0, mask0)]
    for inst_id in inst_ids:
        if inst_id == inst0:
            continue
        frame_id = index.inst_to_frames[inst_id][0]
        mask = load_mask(index.ann_dir / inst_id / f"{frame_id:05d}.png")
        prompts.append((inst_to_oid[inst_id], frame_id, mask))
    return t0, mask0, prompts, inst_to_oid


@torch.inference_mode()
def propagate_video(
    predictor,
    index: SAVJPEGIndex,
    prompt_idx: int,
    prompt_mask: np.ndarray,
    extra_prompts: List[Tuple[int, int, np.ndarray]],
    sis_config=None,
) -> Dict[int, Dict[int, np.ndarray]]:
    state = predictor.init_state(
        str(index.img_dir),
        offload_video_to_cpu=True,
        offload_state_to_cpu=True,
    )
    predictor.add_new_mask(state, frame_idx=prompt_idx, obj_id=1, mask=(prompt_mask > 0))

    for obj_id, frame_idx, mask in extra_prompts:
        if obj_id == 1 and frame_idx == prompt_idx:
            continue
        try:
            predictor.add_new_mask(state, frame_idx=frame_idx, obj_id=obj_id, mask=(mask > 0))
        except Exception:
            ys, xs = np.where(mask > 0)
            if len(xs) == 0:
                continue
            box = np.array([[xs.min(), ys.min(), xs.max(), ys.max()]], dtype=np.float32)
            predictor.add_new_points_or_box(state, frame_idx=frame_idx, obj_id=obj_id, box=box)

    outputs: Dict[int, Dict[int, np.ndarray]] = {}
    for frame_idx, obj_ids, masks in predictor.propagate_in_video(state):
        maybe_apply_sis(state, sis_config)
        per_obj: Dict[int, np.ndarray] = {}
        for obj_id, mask in zip(obj_ids, masks):
            mask_np = mask.detach().cpu().numpy() if isinstance(mask, torch.Tensor) else np.array(mask)
            mask_np = np.squeeze(mask_np)
            if mask_np.ndim > 2:
                mask_np = mask_np.max(axis=tuple(range(mask_np.ndim - 2)))
            if np.issubdtype(mask_np.dtype, np.floating):
                mask_np = mask_np > 0.5
            per_obj[int(obj_id)] = mask_np.astype(np.uint8)
        outputs[int(frame_idx)] = per_obj
    return outputs


@torch.inference_mode()
def calibrate_sample(
    index: SAVJPEGIndex,
    predictor,
    max_frames: int | None,
    cali_type: str | None,
    sis_config=None,
) -> None:
    prompt_idx, prompt_mask, prompts, _ = collect_prompts(index)
    state = predictor.init_state(
        str(index.img_dir),
        offload_video_to_cpu=True,
        offload_state_to_cpu=True,
    )
    for obj_id, frame_idx, mask in prompts:
        predictor.add_new_mask(state, frame_idx=frame_idx, obj_id=obj_id, mask=(mask > 0))

    for i, _ in enumerate(predictor.propagate_in_video(state)):
        maybe_apply_sis(state, sis_config)
        if i == 0 and cali_type is not None:
            enable_calibration_woquantization(predictor, quantizer_type=cali_type)
        if max_frames is not None and i + 1 >= max_frames:
            break


@torch.inference_mode()
def forward_sample_distribution(
    predictor,
    index: SAVJPEGIndex,
    eps: float,
    max_frames: int | None,
    sis_config=None,
) -> torch.Tensor | None:
    _, _, prompts, _ = collect_prompts(index)
    state = predictor.init_state(
        str(index.img_dir),
        offload_video_to_cpu=True,
        offload_state_to_cpu=True,
    )
    for obj_id, frame_idx, mask in prompts:
        predictor.add_new_mask(state, frame_idx=frame_idx, obj_id=obj_id, mask=(mask > 0))

    values: list[torch.Tensor] = []
    for i, (_frame_idx, _obj_ids, masks) in enumerate(predictor.propagate_in_video(state)):
        maybe_apply_sis(state, sis_config)
        if torch.is_tensor(masks) and masks.numel() > 0:
            values.append(torch.sigmoid(masks.detach()).flatten().to("cpu", dtype=torch.float32))
        if max_frames is not None and i + 1 >= max_frames:
            break

    if not values:
        return None
    distribution = torch.cat(values)
    total = distribution.sum()
    if not torch.isfinite(total) or total <= eps:
        return None
    return (distribution / total.clamp_min(eps)).clamp_min(eps)


def calibrate(
    dataset: SAVVideoDataset,
    predictor,
    num_sample: int,
    num_frame: int | None,
    seed: int,
    cali_type: str | None,
    sis_config=None,
) -> None:
    rng = random.Random(seed)
    indices = list(range(len(dataset)))
    rng.shuffle(indices)
    indices = indices[: min(num_sample, len(indices))]

    print(f"=== Calibration with {len(indices)} videos ===")
    progress = tqdm(indices, desc="Calibrating", unit="video")
    for idx in progress:
        sample = dataset[idx]
        progress.set_postfix(sample=sample.sample)
        calibrate_sample(
            sample,
            predictor,
            max_frames=num_frame,
            cali_type=cali_type,
            sis_config=sis_config,
        )


def calibrate_quantized_model(
    predictor,
    train_dataset: SAVVideoDataset,
    q_config,
    seed: int,
    sis_config=None,
) -> None:
    num_sample = min(int(q_config.calibrate.sample), len(train_dataset))
    num_frame = int(q_config.calibrate.frame) if q_config.calibrate.frame else None
    act_num_frame = num_frame + 1 if num_frame is not None else None
    disable_all(predictor)

    if q_config.ptq4sam.BIG:
        calibrate(train_dataset, predictor, 1, 2, seed, cali_type=None, sis_config=sis_config)
        bimodal_adjust(predictor)
        disable_all(predictor)

    enable_calibration_woquantization(predictor, quantizer_type="act_fake_quant")
    calibrate(
        train_dataset,
        predictor,
        num_sample,
        act_num_frame,
        seed,
        cali_type="act_fake_quant",
        sis_config=sis_config,
    )
    disable_all(predictor)

    enable_calibration_woquantization(predictor, quantizer_type="weight_fake_quant")
    calibrate(
        train_dataset,
        predictor,
        1,
        2,
        seed,
        cali_type="weight_fake_quant",
        sis_config=sis_config,
    )
    disable_all(predictor)


def recon_model(predictor, fp_predictor, train_dataset: SAVVideoDataset, q_config) -> None:
    def visit(module, fp_module):
        for name, child in module.named_children():
            fp_child = getattr(fp_module, name)
            if isinstance(child, (QuantizedLayer, QuantizedBlock, PreQuantizedLayer, QuantizedMatMul)):
                print(f"Start reconstruction for module:\n{child}")
                reconstruction(predictor, fp_predictor, child, fp_child, train_dataset, q_config)
            else:
                visit(child, fp_child)

    visit(predictor, fp_predictor)


def predict_sample(
    index: SAVJPEGIndex,
    predictor,
    pred_root: Path | None,
    skip_first_last: bool,
    sis_config=None,
) -> Tuple[List[float], List[float]]:
    prompt_idx, prompt_mask, prompts, inst_to_oid = collect_prompts(index)
    outputs = propagate_video(
        predictor=predictor,
        index=index,
        prompt_idx=prompt_idx,
        prompt_mask=prompt_mask,
        extra_prompts=prompts,
        sis_config=sis_config,
    )

    j_scores: List[float] = []
    f_scores: List[float] = []
    for inst_id in sorted(index.inst_to_frames.keys()):
        obj_id = inst_to_oid[inst_id]
        frames = index.inst_to_frames[inst_id]
        if skip_first_last and len(frames) >= 3:
            frames = frames[1:-1]
        for frame_id in frames:
            pred = outputs.get(frame_id, {}).get(obj_id)
            if pred is None:
                pred = np.zeros((index.H, index.W), dtype=np.uint8)
            gt = load_mask(index.ann_dir / inst_id / f"{frame_id:05d}.png")
            j_scores.append(iou(pred, gt))
            f_scores.append(f_measure(pred, gt))
            if pred_root is not None:
                save_mask(pred_root / index.sample / inst_id / f"{frame_id:05d}.png", pred)
    return j_scores, f_scores


def evaluate_dataset(
    predictor,
    dataset: SAVVideoDataset,
    pred_root: Path | None,
    sis_config=None,
) -> dict[str, float | int]:
    all_j: List[float] = []
    all_f: List[float] = []
    progress = tqdm(range(len(dataset)), desc="Evaluating", unit="video")
    for idx in progress:
        index = dataset[idx]
        progress.set_postfix(sample=index.sample)
        j_scores, f_scores = predict_sample(
            index=index,
            predictor=predictor,
            pred_root=pred_root,
            skip_first_last=True,
            sis_config=sis_config,
        )
        all_j.extend(j_scores)
        all_f.extend(f_scores)
        if all_j:
            progress.set_postfix(
                sample=index.sample,
                J=f"{np.mean(all_j) * 100:.2f}",
                F=f"{np.mean(all_f) * 100:.2f}",
            )

    mean_j = float(np.mean(all_j)) if all_j else 0.0
    mean_f = float(np.mean(all_f)) if all_f else 0.0
    return {
        "frames": len(all_j),
        "mean_J": mean_j,
        "mean_F": mean_f,
        "mean_JF": 0.5 * (mean_j + mean_f),
    }


def main() -> None:
    args = parse_args()
    train_dataset = build_dataset(args.sav_train)
    test_dataset = build_dataset(args.sav_test)

    predictor = build_predictor(args)
    bit_allocation = None
    sis_config = None
    reset_sis_stats()
    q_config = None
    if args.q_config:
        q_config = normalize_quant_config(quant_utils.parse_config(args.q_config))
        sis_config = getattr(getattr(q_config, "mix_qsam2", None), "sis", None)

    if not args.fp_model:
        if q_config is None:
            raise ValueError("--q_config is required for quantized Mix-QSAM2 evaluation.")
        print_quant_settings(q_config)

        predictor = quantize_model(predictor, q_config)
        predictor.to(args.device)
        predictor.eval()
        set_observer_names(predictor)

        disable_all(predictor)
        bit_allocation = search_mixed_precision_bits(
            model=predictor,
            train_dataset=train_dataset,
            q_config=q_config,
            forward_distribution=forward_sample_distribution,
            seed=args.calib_seed,
            sis_config=sis_config,
        )
        apply_mixed_precision_bits(predictor, bit_allocation)
        if args.save_bit_allocation is not None and bit_allocation is not None:
            args.save_bit_allocation.parent.mkdir(parents=True, exist_ok=True)
            with args.save_bit_allocation.open("w") as f:
                json.dump(bit_allocation, f, indent=2)
            print(f"Saved bit allocation to: {args.save_bit_allocation}")

        fp_predictor = None
        if args.recon:
            fp_predictor = deepcopy(predictor)
            disable_all(fp_predictor)

        calibrate_quantized_model(
            predictor=predictor,
            train_dataset=train_dataset,
            q_config=q_config,
            seed=args.calib_seed,
            sis_config=sis_config,
        )
        enable_quantization(predictor)

        if args.recon:
            print("Begin Model Reconstruction...")
            if fp_predictor is None:
                fp_predictor = deepcopy(predictor)
                disable_all(fp_predictor)
            recon_model(predictor, fp_predictor, train_dataset, q_config)
            enable_quantization(predictor)
            for module in predictor.modules():
                if hasattr(module, "drop_prob"):
                    module.drop_prob = 1

    start = time.time()
    summary = evaluate_dataset(predictor, test_dataset, args.pred_root, sis_config=sis_config)
    summary.update(
        {
            "elapsed_sec": time.time() - start,
            "model_cfg": str(args.model_cfg),
            "checkpoint": str(args.checkpoint),
            "sav_train": str(args.sav_train),
            "sav_test": str(args.sav_test),
            "fp_model": bool(args.fp_model),
            "quantized": not bool(args.fp_model),
            "recon": bool(args.recon) and not bool(args.fp_model),
            "mix_qsam2_allocation": bit_allocation is not None,
            "sis": bool(getattr(sis_config, "enabled", False)) if sis_config is not None else False,
            "sis_stats": get_sis_stats(),
        }
    )

    print("========== Mix-QSAM2 Video Evaluation (SA-V) ==========")
    for key, value in summary.items():
        if isinstance(value, float):
            print(f"{key}: {value:.6f}")
        else:
            print(f"{key}: {value}")

    if args.save_json is not None:
        args.save_json.parent.mkdir(parents=True, exist_ok=True)
        with args.save_json.open("w") as f:
            json.dump({"summary": summary}, f, indent=2)
        print(f"Saved results to: {args.save_json}")


if __name__ == "__main__":
    main()
