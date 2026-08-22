from __future__ import annotations

import argparse
import importlib
import json
import os
import os.path as osp
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from mmcv import Config, DictAction
from mmcv.cnn import fuse_conv_bn
from mmcv.runner import wrap_fp16_model
from mmdet.apis import single_gpu_test
from mmdet.datasets import build_dataloader, build_dataset, replace_ImageToTensor
from mmdet.models import build_detector
from mmdet.utils import compat_cfg, get_device, replace_cfg_vals, setup_multi_processes, update_data_root

BASELINE_ROOT = Path(__file__).resolve().parent
BASELINES_ROOT = BASELINE_ROOT.parent
REPO_ROOT = BASELINES_ROOT.parent
SAM_ROOT = REPO_ROOT / "sam"
SOLVER_ROOT = SAM_ROOT / "ahcqsam" / "solver"
MMDET_ROOT = SAM_ROOT / "mmdetection"
for path in (BASELINES_ROOT, SAM_ROOT, SOLVER_ROOT, MMDET_ROOT):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from ahcqsam.model.quant_model import bimodal_adjust, specials  # noqa: E402
from ahcqsam.quantization.observer import ObserverBase  # noqa: E402
from ahcqsam.quantization.quantized_module import QuantizedLayer  # noqa: E402
from ahcqsam.quantization.state import (  # noqa: E402
    enable_calibration_woquantization,
    enable_quantization,
)
import utils  # noqa: E402


class PTQWrapperConfig:
    def __getattr__(self, _name):
        return False


PTQ_WRAPPER_CONFIG = PTQWrapperConfig()


class PrintLogger:
    def info(self, message, *args):
        print(message % args if args else message)


PRINT_LOGGER = PrintLogger()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate SAM with a Mix-QSAM-ready PTQ entry.")
    parser.add_argument("--config", default="../sam/projects/configs/yolox/yolo_l-sam-vit-b.py", help="Detector + SAM config file.")
    parser.add_argument("--mix-config", default="./mix_qsam/exp/config66.yaml", help="Mix-QSAM/PTQ config file.")
    parser.add_argument("--fp-model", action="store_true", default=False)
    parser.add_argument("--save-bit-allocation", default=None, help="Optional path to save the searched bit allocation JSON.")
    parser.add_argument("--quant-encoder", action="store_true", default=False)
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--fuse-conv-bn", action="store_true", default=False)
    parser.add_argument("--eval", type=str, nargs="+", default=["segm"])
    parser.add_argument("--eval-options", nargs="+", action=DictAction)
    parser.add_argument("--cfg-options", nargs="+", action=DictAction)
    return parser.parse_args()


def import_config_plugins(cfg: Config, config_path: str) -> None:
    if not getattr(cfg, "plugin", False):
        return

    if hasattr(cfg, "plugin_dir"):
        module_dir = cfg.plugin_dir
    else:
        module_dir = osp.dirname(config_path)

    module_path = module_dir.replace("/", ".").rstrip(".")
    importlib.import_module(module_path)


def load_mmdet_config(args: argparse.Namespace) -> Config:
    cfg = Config.fromfile(args.config)
    cfg = replace_cfg_vals(cfg)
    update_data_root(cfg)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    cfg = compat_cfg(cfg)
    # Config files were authored for execution from sam/. Anchor their model assets there.
    if cfg.model.get("sam_checkpoint"):
        cfg.model.sam_checkpoint = str(SAM_ROOT / cfg.model.sam_checkpoint)
    if cfg.model.get("det_model_ckpt"):
        cfg.model.det_model_ckpt = str(SAM_ROOT / cfg.model.det_model_ckpt)
    det_cfg = cfg.model.get("det_wrapper_cfg")
    if det_cfg:
        if det_cfg.get("det_config"):
            det_cfg.det_config = str(SAM_ROOT / det_cfg.det_config)
        if det_cfg.get("det_weight"):
            det_cfg.det_weight = str(SAM_ROOT / det_cfg.det_weight)
    setup_multi_processes(cfg)
    import_config_plugins(cfg, args.config)

    if cfg.get("cudnn_benchmark", False):
        torch.backends.cudnn.benchmark = True

    if "pretrained" in cfg.model:
        cfg.model.pretrained = None
    elif cfg.model.get("backbone", None) is not None and "init_cfg" in cfg.model.backbone:
        cfg.model.backbone.init_cfg = None

    cfg.gpu_ids = [args.gpu_id]
    cfg.device = get_device()
    return cfg


def build_test_dataset_and_loader(cfg: Config):
    if isinstance(cfg.data.test, dict):
        cfg.data.test.test_mode = True
        if cfg.data.test_dataloader.get("samples_per_gpu", 1) > 1:
            cfg.data.test.pipeline = replace_ImageToTensor(cfg.data.test.pipeline)
    elif isinstance(cfg.data.test, list):
        for ds_cfg in cfg.data.test:
            ds_cfg.test_mode = True
        if cfg.data.test_dataloader.get("samples_per_gpu", 1) > 1:
            for ds_cfg in cfg.data.test:
                ds_cfg.pipeline = replace_ImageToTensor(ds_cfg.pipeline)

    test_loader_cfg = {
        "samples_per_gpu": 1,
        "workers_per_gpu": 2,
        "dist": False,
        "shuffle": False,
        **cfg.data.get("test_dataloader", {}),
    }
    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(dataset, **test_loader_cfg)
    return dataset, data_loader


def quantize_model(model: nn.Module, q_config, quant_encoder: bool) -> nn.Module:
    def replace_module(module, w_qconfig, a_qconfig, ptq4sam_config, qoutput=True):
        prev_quantmodule = None
        for name, child_module in module.named_children():
            if (
                "patch_embed" in name
                or "output_upscaling" in name
                or "iou_prediction_head" in name
                or "output_hypernetworks_mlps" in name
            ):
                continue
            if type(child_module) in specials:
                setattr(
                    module,
                    name,
                    specials[type(child_module)](
                        child_module,
                        w_qconfig,
                        a_qconfig,
                        PTQ_WRAPPER_CONFIG,
                        ptq4sam_config,
                    ),
                )
            elif isinstance(child_module, (nn.Conv2d, nn.Linear)):
                setattr(
                    module,
                    name,
                    QuantizedLayer(child_module, None, w_qconfig, a_qconfig, qoutput),
                )
                prev_quantmodule = getattr(module, name)
            elif isinstance(child_module, (nn.ReLU, nn.ReLU6, nn.GELU)):
                if prev_quantmodule is not None:
                    prev_quantmodule.activation = child_module
                    setattr(module, name, nn.Identity())
            elif not isinstance(child_module, nn.Identity):
                replace_module(
                    child_module,
                    w_qconfig,
                    a_qconfig,
                    ptq4sam_config,
                    qoutput,
                )

    if quant_encoder:
        model.predictor.model.image_encoder = specials[type(model.predictor.model.image_encoder)](
            model.predictor.model.image_encoder,
            q_config.w_qconfig,
            q_config.a_qconfig,
            PTQ_WRAPPER_CONFIG,
            q_config.ptq4sam,
        )

    replace_module(
        model.predictor.model.mask_decoder,
        q_config.w_qconfig,
        q_config.a_qconfig,
        q_config.ptq4sam,
    )
    return model


def set_observer_names(model: nn.Module) -> None:
    for name, module in model.named_modules():
        if isinstance(module, ObserverBase):
            module.set_name(name)


def apply_mixed_precision_bits(model: nn.Module, q_config, allocation: dict | None = None) -> None:
    mix_cfg = getattr(q_config, "mix_qsam", None)
    if mix_cfg is None or not mix_cfg.enabled:
        return
    allocation_source = "in-memory allocation"
    if allocation is None:
        raise ValueError("mix_qsam.enabled requires an in-memory allocation.")

    applied = 0
    skipped = []
    for unit in allocation.get("units", []):
        bit = int(unit["bit"])
        targets = allocation_unit_to_quantizers(unit["name"])
        if not targets:
            skipped.append((unit["name"], "no quantizer in current PTQ wrapper"))
            continue
        unit_applied = 0
        for target in targets:
            quantizer = get_nested_module(model.predictor.model, target)
            if quantizer is None or not hasattr(quantizer, "set_bit"):
                skipped.append((unit["name"], target))
                continue
            quantizer.set_bit(bit)
            unit_applied += 1
        applied += unit_applied

    print(f"Applied Mix-QSAM bits from {allocation_source}")
    print(f"  quantizers updated: {applied}")
    print(f"  allocation units skipped: {len(skipped)}")
    if skipped:
        print("  skipped examples:")
        for unit_name, target in skipped[:20]:
            print(f"    {unit_name} -> {target}")


def get_nested_module(root: nn.Module, path: str):
    module = root
    for part in path.split("."):
        if part.isdigit():
            part = int(part)
            try:
                module = module[part]
            except (TypeError, IndexError):
                return None
        else:
            if not hasattr(module, part):
                return None
            module = getattr(module, part)
    return module


def allocation_unit_to_quantizers(unit_name: str) -> list[str]:
    parts = unit_name.split(".")
    if unit_name.startswith("image_encoder.neck."):
        return prequantized_layer_quantizers(unit_name.replace("image_encoder.neck.", "image_encoder.neck.model."))

    if ".attn." in unit_name and unit_name.startswith("image_encoder."):
        prefix, op = unit_name.rsplit(".", 1)
        if op in {"qkv"}:
            return prequantized_layer_quantizers(unit_name)
        if op in {"q", "k", "v"}:
            return [f"{prefix}.{op}_post_act_fake_quantize"]
        if op == "softmax":
            return [f"{prefix}.softmax_post_act_fake_quantize"]
        if op == "matmul2":
            proj_prefix = unit_name.rsplit(".", 1)[0] + ".proj"
            return [f"{proj_prefix}.layer_pre_act_fake_quantize"]
        if op == "proj":
            return [f"{unit_name}.module.weight_fake_quant"]
        return []

    if unit_name.startswith("mask_decoder.transformer.") and ".mlp." in unit_name:
        return prequantized_layer_quantizers(unit_name)

    if unit_name.startswith("image_encoder.") and ".mlp." in unit_name:
        return prequantized_layer_quantizers(unit_name)

    if unit_name.startswith("mask_decoder.transformer."):
        prefix, op = unit_name.rsplit(".", 1)
        if op in {"q_proj", "k_proj", "v_proj"}:
            return prequantized_layer_quantizers(unit_name)
        if op in {"q", "k", "v"}:
            return [f"{prefix}.{op}_post_act_fake_quantize"]
        if op == "softmax":
            return [f"{prefix}.softmax_post_act_fake_quantize"]
        if op == "matmul2":
            out_prefix = unit_name.rsplit(".", 1)[0] + ".out_proj"
            return [f"{out_prefix}.layer_pre_act_fake_quantize"]
        if op == "out_proj":
            return [f"{unit_name}.module.weight_fake_quant"]
        return []

    if parts[-1] in {"lin1", "lin2"}:
        return prequantized_layer_quantizers(unit_name)
    return []


def prequantized_layer_quantizers(prefix: str) -> list[str]:
    return [
        f"{prefix}.module.weight_fake_quant",
        f"{prefix}.layer_pre_act_fake_quantize",
    ]


@torch.no_grad()
def search_mixed_precision_bits(
    model: nn.Module,
    cfg: Config,
    q_config,
    args: argparse.Namespace,
) -> dict | None:
    mix_cfg = getattr(q_config, "mix_qsam", None)
    if mix_cfg is None or not mix_cfg.enabled:
        return None

    from compute_allocation import (  # noqa: WPS433
        cap_detector_boxes,
        discover_layer_units,
        load_calibration_cpu,
        load_or_collect_scores,
        solve_bit_allocation,
    )

    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
    score_samples = int(mix_cfg.score_samples)
    max_boxes_per_image = int(mix_cfg.max_boxes_per_image)
    target_bits = int(mix_cfg.target_bits)
    lambda_btr = float(mix_cfg.lambda_btr)
    eps = float(mix_cfg.eps)
    score_cache = mix_cfg.score_cache
    allocation_output = args.save_bit_allocation

    print("Search Mix-QSAM bit allocation")
    print(f"  target_bits: {target_bits}")
    print(f"  candidate_bits: {list(mix_cfg.candidate_bits)}")
    print(f"  score_samples: {score_samples}")
    print(f"  max_boxes_per_image: {max_boxes_per_image}")
    print(f"  solver: {mix_cfg.solver}:{mix_cfg.cvxpy_solver}")
    print(f"  use_bitops: {mix_cfg.use_bitops}")
    print(f"  min_budget_ratio: {mix_cfg.min_budget_ratio}")

    original_simple_test = model.det_model.simple_test
    cap_detector_boxes(model, max_boxes_per_image)
    try:
        score_data = load_calibration_cpu(cfg, num_samples=score_samples)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        units = discover_layer_units(model)
        print(f"Discovered {len(units)} Mix-QSAM layer units.")
        scores = load_or_collect_scores(model, score_data, units, eps, score_cache, device)
    finally:
        model.det_model.simple_test = original_simple_test

    bits, allocation_metadata = solve_bit_allocation(
        units=units,
        importance=scores["importance"],
        synergy=scores["synergy"],
        candidate_bits=list(mix_cfg.candidate_bits),
        target_bits=target_bits,
        lambda_btr=lambda_btr,
        solver=mix_cfg.solver,
        use_bitops=bool(mix_cfg.use_bitops),
        cvxpy_solver=mix_cfg.cvxpy_solver,
        cvxpy_verbose=bool(mix_cfg.cvxpy_verbose),
        solver_time_limit=float(mix_cfg.solver_time_limit),
        min_budget_ratio=float(mix_cfg.min_budget_ratio),
    )

    allocation_metadata.setdefault(
        "used_budget_bitops",
        sum(unit.macs * bit * bit for unit, bit in zip(units, bits)),
    )
    allocation_metadata.setdefault(
        "target_budget_bitops",
        sum(unit.macs for unit in units) * target_bits * target_bits,
    )

    allocation = {
        "config": {
            "detector_config": args.config,
            "mix_config": args.mix_config,
            "score_samples": score_samples,
            "max_boxes_per_image": max_boxes_per_image,
            "target_bits": target_bits,
            "candidate_bits": list(mix_cfg.candidate_bits),
            "lambda_btr": lambda_btr,
            "solver": mix_cfg.solver,
            "cvxpy_solver": mix_cfg.cvxpy_solver,
            "cvxpy_verbose": bool(mix_cfg.cvxpy_verbose),
            "solver_time_limit": float(mix_cfg.solver_time_limit),
            "use_bitops": bool(mix_cfg.use_bitops),
            "min_budget_ratio": float(mix_cfg.min_budget_ratio),
            "eps": eps,
        },
        "score_summary": {
            "num_images": scores["num_images"],
            "skipped_images": scores["skipped_images"],
        },
        "allocation": allocation_metadata,
        "units": [
            {
                "name": unit.name,
                "type": unit.module_type,
                "num_params": unit.num_params,
                "macs": unit.macs,
                "importance": scores["importance"][idx],
                "bit": bits[idx],
            }
            for idx, unit in enumerate(units)
        ],
        "adjacent_synergy": [
            {
                "left": units[idx].name,
                "right": units[idx + 1].name,
                "synergy": score,
            }
            for idx, score in enumerate(scores["synergy"])
        ],
        "per_image_scores": scores["per_image_scores"],
    }

    if allocation_output:
        Path(allocation_output).parent.mkdir(parents=True, exist_ok=True)
        with open(allocation_output, "w") as f:
            json.dump(allocation, f, indent=2)
        print(f"Saved Mix-QSAM bit allocation to {allocation_output}")
    else:
        print("Skip saving Mix-QSAM bit allocation JSON.")

    return allocation


@torch.no_grad()
def calibrate(model: nn.Module, cali_data, q_config) -> None:
    start = time.time()
    if q_config.ptq4sam.BIG:
        model.extract_feat(cali_data[0])
        bimodal_adjust(model, logger=PRINT_LOGGER)

    enable_calibration_woquantization(model, quantizer_type="act_fake_quant")
    for data in cali_data:
        model.extract_feat(data)

    enable_calibration_woquantization(model, quantizer_type="weight_fake_quant")
    model.extract_feat(cali_data[0])

    print(f"Calibration time: {time.time() - start:.2f} sec")


def build_model(cfg: Config, dataset, args: argparse.Namespace) -> nn.Module:
    cfg.model.train_cfg = None
    model = build_detector(cfg.model, test_cfg=cfg.get("test_cfg"))
    fp16_cfg = cfg.get("fp16", None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    if args.fuse_conv_bn:
        model = fuse_conv_bn(model)
    model.CLASSES = dataset.CLASSES
    return model


def main() -> None:
    args = parse_args()
    cfg = load_mmdet_config(args)
    q_config = utils.parse_config(args.mix_config)

    dataset, data_loader = build_test_dataset_and_loader(cfg)
    model = build_model(cfg, dataset, args)

    if not args.fp_model:
        print("Run quantized SAM")
        print(f"quant_encoder: {args.quant_encoder}")
        print(f"activation quantizer: {q_config.a_qconfig.quantizer} / {q_config.a_qconfig.bit}-bit")
        print(f"weight quantizer: {q_config.w_qconfig.quantizer} / {q_config.w_qconfig.bit}-bit")
        print(f"mix_qsam.enabled: {q_config.mix_qsam.enabled}")

        for parameter in model.parameters():
            parameter.requires_grad = False
        model.cuda()
        model.eval()
        bit_allocation = search_mixed_precision_bits(model, cfg, q_config, args)

        model = quantize_model(model, q_config, args.quant_encoder)
        model.eval()
        set_observer_names(model)
        apply_mixed_precision_bits(model, q_config, bit_allocation)

        cali_data = utils.load_calibration(
            cfg,
            distributed=False,
            num_samples=q_config.calibrate.sample,
        )
        calibrate(model, cali_data, q_config)
        enable_quantization(model)

        for _, module in model.named_modules():
            if hasattr(module, "drop_prob"):
                module.drop_prob = 1

    model.det_model.cuda()
    model = build_detector_parallel(model, cfg, args)
    outputs = single_gpu_test(model, data_loader, show=False)

    eval_kwargs = cfg.get("evaluation", {}).copy()
    for key in ["interval", "tmpdir", "start", "gpu_collect", "save_best", "rule", "dynamic_intervals"]:
        eval_kwargs.pop(key, None)
    if args.eval_options is not None:
        eval_kwargs.update(args.eval_options)
    eval_kwargs.update(dict(metric=args.eval))

    metric = dataset.evaluate(outputs, **eval_kwargs)
    print(metric)


def build_detector_parallel(model: nn.Module, cfg: Config, args: argparse.Namespace) -> nn.Module:
    from mmdet.utils import build_dp

    return build_dp(model, cfg.device, device_ids=cfg.gpu_ids)


if __name__ == "__main__":
    main()
