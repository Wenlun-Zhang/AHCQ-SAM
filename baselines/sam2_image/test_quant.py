from __future__ import annotations

import argparse
from copy import deepcopy
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image
import torch
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

from sam2.build_sam import build_sam2  # noqa: E402
from sam2.sam2_image_predictor import SAM2ImagePredictor  # noqa: E402

from model.quant_model import bimodal_adjust, specials  # noqa: E402
from quantization.observer import ObserverBase  # noqa: E402
from quantization.state import (  # noqa: E402
    disable_all,
    enable_calibration_woquantization,
    enable_quantization,
)
from solver import utils as quant_utils  # noqa: E402

from coco_dataset import CocoBoxPromptDataset, CocoImageRecord  # noqa: E402
from metrics import MeanIoUMeter, mask_iou  # noqa: E402
from recon import condi_based_act_recon_model_image, recon_model_image  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate SAM2 image segmentation on COCO GT box prompts."
    )
    parser.add_argument("--coco-root", type=Path, required=True)
    parser.add_argument("--model-cfg", type=str, default="configs/sam2.1/sam2.1_hiera_t.yaml")
    parser.add_argument("--checkpoint", type=Path, default=SAM2_REPO / "checkpoints" / "sam2.1_hiera_tiny.pt")
    parser.add_argument("--q_config", type=str, default="./sam2_image/exp/config66.yaml")
    parser.add_argument("--fp-model", action="store_true", default=False)
    parser.add_argument("--recon", action="store_true", default=False)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--calib-seed", type=int, default=0)
    parser.add_argument("--min-area", type=float, default=1.0)
    parser.add_argument("--include-crowd", action="store_true", default=False)
    parser.add_argument("--multimask-output", action="store_true", default=False)
    parser.add_argument("--save-json", type=Path, default=None)
    return parser.parse_args()


def coco_split_paths(coco_root: Path, split: str) -> tuple[Path, Path]:
    image_root = coco_root / f"{split}2017"
    ann_file = coco_root / "annotations" / f"instances_{split}2017.json"
    return image_root, ann_file


def build_image_predictor(args: argparse.Namespace) -> SAM2ImagePredictor:
    model = build_sam2(args.model_cfg, str(args.checkpoint), device=args.device)
    model.eval()
    return SAM2ImagePredictor(model)


def quantize_model(model: torch.nn.Module, config_quant) -> torch.nn.Module:
    skipped_roots = {"memory_attention", "memory_encoder"}

    def replace_module(module, w_qconfig, a_qconfig, ahcqsam_config, ptq4sam_config):
        for name, child in module.named_children():
            if name in skipped_roots:
                continue
            if type(child) in specials:
                setattr(
                    module,
                    name,
                    specials[type(child)](
                        child,
                        w_qconfig,
                        a_qconfig,
                        ahcqsam_config,
                        ptq4sam_config,
                    ),
                )
            else:
                replace_module(
                    child,
                    w_qconfig,
                    a_qconfig,
                    ahcqsam_config,
                    ptq4sam_config,
                )

    replace_module(
        model,
        config_quant.w_qconfig,
        config_quant.a_qconfig,
        config_quant.ahcqsam,
        config_quant.ptq4sam,
    )
    return model


def set_observer_names(model: torch.nn.Module) -> None:
    for name, module in model.named_modules():
        if isinstance(module, ObserverBase):
            module.set_name(name)


def print_quant_settings(q_config) -> None:
    print("=== Quantization Settings ===")
    print(f"acnr: {q_config.ahcqsam.acnr}")
    print(f"cag: {q_config.ahcqsam.cag}")
    print(f"hluq: {q_config.ahcqsam.hluq}")
    print(f"lnq: {q_config.ahcqsam.lnq}")
    print(f"BIG: {q_config.ptq4sam.BIG}")
    print(f"AGQ: {q_config.ptq4sam.AGQ}")
    print(f"calibrate.sample: {q_config.calibrate.sample}")
    print(f"recon.sample: {q_config.recon.sample}")
    print(f"recon.batch_size: {q_config.recon.batch_size}")


def select_mask(masks: np.ndarray, scores: np.ndarray) -> np.ndarray:
    if masks.ndim != 3:
        raise ValueError(f"Expected CxHxW masks, got shape {masks.shape}")
    if masks.shape[0] == 1:
        return masks[0]
    return masks[int(np.argmax(scores))]


@torch.inference_mode()
def evaluate_record(
    predictor: SAM2ImagePredictor,
    record: CocoImageRecord,
    multimask_output: bool,
) -> list[tuple[int, float, float]]:
    image = np.array(Image.open(record.image_path).convert("RGB"))
    predictor.set_image(image)

    results: list[tuple[int, float, float]] = []
    for instance in record.instances:
        masks, scores, _ = predictor.predict(
            box=instance.bbox_xyxy,
            multimask_output=multimask_output,
            return_logits=False,
        )
        pred_mask = select_mask(masks, scores)
        iou = mask_iou(pred_mask, instance.gt_mask)
        score = float(np.max(scores)) if len(scores) else float("nan")
        results.append((instance.ann_id, iou, score))
    return results


def calibration_sample_count(q_config) -> int:
    calibrate_config = getattr(q_config, "calibrate", 1)
    if isinstance(calibrate_config, int):
        return calibrate_config
    return int(getattr(calibrate_config, "sample", 1))


def select_calibration_image_ids(
    dataset: CocoBoxPromptDataset,
    num_sample: int,
    seed: int,
) -> list[int]:
    image_ids = list(dataset.image_ids)
    rng = random.Random(seed)
    rng.shuffle(image_ids)
    return image_ids[: min(num_sample, len(image_ids))]


@torch.inference_mode()
def calibration_forward(
    predictor: SAM2ImagePredictor,
    dataset: CocoBoxPromptDataset,
    image_ids: list[int],
    multimask_output: bool,
    desc: str,
) -> int:
    instance_count = 0
    progress = tqdm(
        dataset.iter_image_ids(image_ids),
        total=len(image_ids),
        desc=desc,
        unit="image",
    )
    for record in progress:
        if not record.image_path.exists():
            raise FileNotFoundError(f"Image file not found: {record.image_path}")

        image = np.array(Image.open(record.image_path).convert("RGB"))
        predictor.set_image(image)
        for instance in record.instances:
            predictor.predict(
                box=instance.bbox_xyxy,
                multimask_output=multimask_output,
                return_logits=True,
            )
            instance_count += 1
        progress.set_postfix(instances=instance_count)
    return instance_count


def calibrate_quantized_model(
    predictor: SAM2ImagePredictor,
    fp_predictor: SAM2ImagePredictor | None,
    dataset: CocoBoxPromptDataset,
    q_config,
    multimask_output: bool,
    seed: int,
) -> None:
    num_sample = calibration_sample_count(q_config)
    image_ids = select_calibration_image_ids(dataset, num_sample, seed)
    if not image_ids:
        raise RuntimeError("No images are available for calibration.")

    print(f"=== Calibration with {len(image_ids)} COCO images ===")
    disable_all(predictor.model)

    if q_config.ptq4sam.BIG:
        calibration_forward(
            predictor=predictor,
            dataset=dataset,
            image_ids=image_ids[:1],
            multimask_output=multimask_output,
            desc="BIG calibration",
        )
        bimodal_adjust(predictor.model)

    if q_config.ahcqsam.acnr:
        if fp_predictor is None:
            raise RuntimeError("ACNR requires an FP counterpart predictor.")
        enable_calibration_woquantization(
            predictor.model,
            quantizer_type="act_fake_quant",
        )
        calibration_forward(
            predictor=predictor,
            dataset=dataset,
            image_ids=image_ids,
            multimask_output=multimask_output,
            desc="ACNR activation calibration",
        )
        condi_based_act_recon_model_image(
            predictor=predictor,
            fp_predictor=fp_predictor,
            dataset=dataset,
            q_config=q_config,
            multimask_output=multimask_output,
            seed=seed,
        )
        disable_all(predictor.model)

    enable_calibration_woquantization(
        predictor.model,
        quantizer_type="act_fake_quant",
    )
    act_instances = calibration_forward(
        predictor=predictor,
        dataset=dataset,
        image_ids=image_ids,
        multimask_output=multimask_output,
        desc="Activation calibration",
    )
    if act_instances == 0:
        raise RuntimeError("No valid COCO instances were used for calibration.")

    disable_all(predictor.model)
    enable_calibration_woquantization(
        predictor.model,
        quantizer_type="weight_fake_quant",
    )
    calibration_forward(
        predictor=predictor,
        dataset=dataset,
        image_ids=image_ids[:1],
        multimask_output=multimask_output,
        desc="Weight calibration",
    )
    disable_all(predictor.model)


def main() -> None:
    args = parse_args()
    eval_img_root, eval_ann = coco_split_paths(args.coco_root, "val")

    dataset = CocoBoxPromptDataset(
        image_root=eval_img_root,
        ann_file=eval_ann,
        min_area=args.min_area,
        include_crowd=args.include_crowd,
    )
    predictor = build_image_predictor(args)

    if not args.fp_model:
        if not args.q_config:
            raise ValueError("--q_config is required when running quantized evaluation.")

        q_config = quant_utils.parse_config(args.q_config)
        print_quant_settings(q_config)
        predictor.model = quantize_model(predictor.model, q_config)
        predictor.model.to(args.device)
        predictor.model.eval()
        set_observer_names(predictor.model)
        needs_fp_counterpart = args.recon or q_config.ahcqsam.acnr
        fp_predictor = None
        if needs_fp_counterpart:
            fp_model = deepcopy(predictor.model)
            disable_all(fp_model)
            fp_predictor = SAM2ImagePredictor(fp_model)

        calib_img_root, calib_ann = coco_split_paths(args.coco_root, "train")
        calib_dataset = CocoBoxPromptDataset(
            image_root=calib_img_root,
            ann_file=calib_ann,
            min_area=args.min_area,
            include_crowd=args.include_crowd,
        )
        calibrate_quantized_model(
            predictor=predictor,
            fp_predictor=fp_predictor,
            dataset=calib_dataset,
            q_config=q_config,
            multimask_output=args.multimask_output,
            seed=args.calib_seed,
        )

        if args.recon:
            print("Begin Model Reconstruction...")
            if fp_predictor is None:
                fp_model = deepcopy(predictor.model)
                disable_all(fp_model)
                fp_predictor = SAM2ImagePredictor(fp_model)
            recon_model_image(
                predictor=predictor,
                fp_predictor=fp_predictor,
                dataset=calib_dataset,
                q_config=q_config,
                multimask_output=args.multimask_output,
                seed=args.calib_seed,
            )
        enable_quantization(predictor.model)

    meter = MeanIoUMeter()
    per_instance: list[dict[str, float | int]] = []

    start = time.time()
    image_count = 0

    progress = tqdm(dataset, total=len(dataset), desc="Evaluating", unit="image")
    for image_count, record in enumerate(progress, 1):
        if not record.image_path.exists():
            raise FileNotFoundError(f"Image file not found: {record.image_path}")

        results = evaluate_record(
            predictor=predictor,
            record=record,
            multimask_output=args.multimask_output,
        )
        ann_by_id = {inst.ann_id: inst for inst in record.instances}
        for ann_id, iou, score in results:
            instance = ann_by_id[ann_id]
            meter.update(iou, instance.area)
            per_instance.append(
                {
                    "image_id": record.image_id,
                    "ann_id": ann_id,
                    "category_id": instance.category_id,
                    "area": instance.area,
                    "iou": iou,
                    "score": score,
                }
            )

        summary = meter.summary()
        progress.set_postfix(
            instances=summary["instances"],
            mIoU=f"{summary['mIoU']:.4f}",
        )

    summary = meter.summary()
    summary["images"] = image_count
    summary["elapsed_sec"] = time.time() - start
    summary["model_cfg"] = str(args.model_cfg)
    summary["checkpoint"] = str(args.checkpoint)
    summary["coco_root"] = str(args.coco_root)
    summary["eval_split"] = "val2017"
    summary["calib_split"] = "train2017" if not args.fp_model else None
    summary["fp_model"] = bool(args.fp_model)
    summary["quantized"] = not bool(args.fp_model)
    summary["multimask_output"] = bool(args.multimask_output)

    print("=== COCO GT-Box SAM2 Image Evaluation ===")
    for key, value in summary.items():
        if isinstance(value, float):
            print(f"{key}: {value:.6f}")
        else:
            print(f"{key}: {value}")

    if args.save_json is not None:
        args.save_json.parent.mkdir(parents=True, exist_ok=True)
        with args.save_json.open("w") as f:
            json.dump({"summary": summary, "instances": per_instance}, f, indent=2)
        print(f"Saved results to: {args.save_json}")


if __name__ == "__main__":
    main()
