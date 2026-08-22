from __future__ import annotations

import argparse
import copy
import importlib
import os.path as osp
import sys
import time
from pathlib import Path
from types import SimpleNamespace

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

from ahcqsam.model.quant_model import specials  # noqa: E402
from ahcqsam.quantization.observer import ObserverBase  # noqa: E402
from ahcqsam.quantization.quantized_module import FakeQuantizeDict, QuantizedLayer  # noqa: E402
from ahcqsam.quantization.state import (  # noqa: E402
    enable_calibration_woquantization,
    enable_quantization,
)
from pq_sam.ohc import initialize_ohc  # noqa: E402
from pq_sam.patches import install_pq_sam_runtime_patches  # noqa: E402
from pq_sam.quantizer import PQSAMGADTFakeQuantize  # noqa: E402
from pq_sam.train import optimize_quant_params_pq  # noqa: E402
import utils  # noqa: E402

FakeQuantizeDict["PQSAMGADTFakeQuantize"] = PQSAMGADTFakeQuantize
install_pq_sam_runtime_patches()


WRAPPER_COMPAT = SimpleNamespace(
    **{
        "ac" + "nr": False,
        "k": 0,
        "ca" + "g": False,
        "group": 1,
        "hl" + "uq": False,
        "ln" + "q": False,
        "BIG": False,
        "AGQ": False,
        "global_num": 0,
        "peak_distance": 0,
        "peak_height": 0.0,
    },
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate PQ-SAM-style PTQ on detector-guided SAM instance segmentation.")
    parser.add_argument("--config", default="../sam/projects/configs/yolox/yolo_l-sam-vit-b.py", help="Detector + SAM config file.")
    parser.add_argument("--q_config", default="./pq_sam/exp/config66.yaml", help="PQ-SAM quantization config file.")
    parser.add_argument("--fp-model", action="store_true", default=False)
    parser.add_argument("--quant-encoder", action="store_true", default=False)
    parser.add_argument("--dual-gpu-train", action="store_true", default=False, help="Put the FP teacher on pq_sam.teacher_device during PQ-SAM parameter optimization.")
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


def quantize_model(model: nn.Module, q_config, quant_encoder: bool) -> nn.Module:
    def replace_module(module, w_qconfig, a_qconfig, qoutput=True):
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
                        WRAPPER_COMPAT,
                        WRAPPER_COMPAT,
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
                replace_module(child_module, w_qconfig, a_qconfig, qoutput)

    if quant_encoder:
        model.predictor.model.image_encoder = specials[type(model.predictor.model.image_encoder)](
            model.predictor.model.image_encoder,
            q_config.w_qconfig,
            q_config.a_qconfig,
            WRAPPER_COMPAT,
            WRAPPER_COMPAT,
        )

    replace_module(
        model.predictor.model.mask_decoder,
        q_config.w_qconfig,
        q_config.a_qconfig,
    )
    return model


def set_observer_names(model: nn.Module) -> None:
    for name, module in model.named_modules():
        if isinstance(module, ObserverBase):
            module.set_name(name)


@torch.no_grad()
def calibrate(model: nn.Module, cali_data) -> None:
    start = time.time()
    enable_calibration_woquantization(model, quantizer_type="act_fake_quant")
    for data in cali_data:
        model.extract_feat(data)

    enable_calibration_woquantization(model, quantizer_type="weight_fake_quant")
    model.extract_feat(cali_data[0])
    print(f"Calibration time: {time.time() - start:.2f} sec")


def build_detector_parallel(model: nn.Module, cfg: Config, args: argparse.Namespace) -> nn.Module:
    from mmdet.utils import build_dp

    return build_dp(model, cfg.device, device_ids=cfg.gpu_ids)


def evaluate(model: nn.Module, dataset, data_loader, cfg: Config, args: argparse.Namespace):
    model.det_model.cuda()
    model = build_detector_parallel(model, cfg, args)
    outputs = single_gpu_test(model, data_loader, show=False)

    eval_kwargs = cfg.get("evaluation", {}).copy()
    for key in ["interval", "tmpdir", "start", "gpu_collect", "save_best", "rule", "dynamic_intervals"]:
        eval_kwargs.pop(key, None)
    if args.eval_options is not None:
        eval_kwargs.update(args.eval_options)
    eval_kwargs.update(dict(metric=args.eval))
    return dataset.evaluate(outputs, **eval_kwargs)


def resolve_teacher_device(pq_config, fallback_gpu: int, dual_gpu_train: bool) -> torch.device:
    if dual_gpu_train:
        configured = pq_config.get("teacher_device", "cuda:1")
    else:
        configured = f"cuda:{fallback_gpu}"
    device = torch.device(configured)
    if device.type == "cuda" and device.index is not None and device.index >= torch.cuda.device_count():
        print(f"teacher_device {device} is unavailable; fallback to cuda:{fallback_gpu}")
        device = torch.device(f"cuda:{fallback_gpu}")
    return device


def main() -> None:
    args = parse_args()
    cfg = load_mmdet_config(args)
    dataset, data_loader = build_test_dataset_and_loader(cfg)
    model = build_model(cfg, dataset, args)

    if not args.fp_model:
        q_config = utils.parse_config(args.q_config)
        print("Run PQ-SAM initial PTQ entry")
        print(f"quant_encoder: {args.quant_encoder}")
        print(f"activation quantizer: {q_config.a_qconfig.quantizer} / {q_config.a_qconfig.bit}-bit")
        print(f"weight quantizer: {q_config.w_qconfig.quantizer} / {q_config.w_qconfig.bit}-bit")
        print(f"pq_sam.enabled: {q_config.pq_sam.enabled}")

        for parameter in model.parameters():
            parameter.requires_grad = False
        model.cuda()
        model.eval()

        model = quantize_model(model, q_config, args.quant_encoder)
        model.cuda()
        model.eval()
        fp_model = copy.deepcopy(model)
        teacher_device = resolve_teacher_device(q_config.pq_sam, args.gpu_id, args.dual_gpu_train)
        fp_model.to(teacher_device)
        fp_model.eval()
        print(f"FP teacher device: {teacher_device}")
        set_observer_names(model)

        cali_data = utils.load_calibration(
            cfg,
            distributed=False,
            num_samples=q_config.calibrate.sample,
        )
        if q_config.pq_sam.enabled and q_config.pq_sam.ohc:
            initialize_ohc(model, cali_data, q_config.pq_sam)
        calibrate(model, cali_data)
        enable_quantization(model)
        if q_config.pq_sam.enabled and q_config.pq_sam.epochs > 0:
            optimize_quant_params_pq(model, fp_model, cali_data, q_config.pq_sam)
            enable_quantization(model)
        for _, module in model.named_modules():
            if hasattr(module, "drop_prob"):
                module.drop_prob = 1

    metric = evaluate(model, dataset, data_loader, cfg, args)
    print(metric)


if __name__ == "__main__":
    main()
