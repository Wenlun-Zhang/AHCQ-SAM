from __future__ import annotations
import sys
sys.path.append("/home/zhang/Project/AHCQ-SAM/sam2/ahcqsam")
import argparse
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
from PIL import Image
import random
import torch
from scipy.ndimage import binary_erosion, binary_dilation
from sam2.build_sam import build_sam2_video_predictor
import utils
from copy import deepcopy
from model.quant_model import bimodal_adjust
from quantization.state import enable_calibration_woquantization, enable_quantization, disable_all
from model.quant_model import specials
from quantization.observer import ObserverBase
from quantization.quantized_module import QuantizedLayer, QuantizedBlock, PreQuantizedLayer, QuantizedMatMul
from solver.recon import reconstruction, condiBasedAct_reconstruction


# =========================
# Tools
# =========================

def read_list(txt: Path) -> List[str]:
    return [x.strip() for x in txt.read_text().splitlines() if x.strip()]


def save_mask(path: Path, mask: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((mask > 0).astype(np.uint8) * 255, mode="L").save(path)


def _boundary(mask: np.ndarray, dilation_ratio: float = 0.02) -> np.ndarray:
    h, w = mask.shape
    diag = float(np.hypot(h, w))
    it = max(1, int(round(dilation_ratio * diag)))
    m = mask > 0
    er = binary_erosion(m, iterations=it)
    dl = binary_dilation(m, iterations=it)
    return np.logical_xor(dl, er)


def iou(pred: np.ndarray, gt: np.ndarray | None) -> float:
    if gt is None:
        return np.nan
    p = pred > 0
    g = gt > 0
    u = np.logical_or(p, g).sum()
    i = np.logical_and(p, g).sum()
    return (i / u) if u > 0 else (1.0 if i == 0 else 0.0)


def f_measure(pred: np.ndarray, gt: np.ndarray | None, eps: float = 1e-6) -> float:
    if gt is None:
        return np.nan
    pb = _boundary(pred)
    gb = _boundary(gt)
    inter = np.logical_and(pb, gb).sum()
    pr = (inter + eps) / (pb.sum() + eps)
    rc = (inter + eps) / (gb.sum() + eps)
    return (2 * pr * rc) / (pr + rc) if (pr + rc) > 0 else 0.0


# =========================
# Video Sample Index: SAVJPEGIndex
# =========================

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

        self.frame_paths: List[Path] = sorted(list(self.img_dir.glob("*.jpg")))
        if not self.frame_paths:
            # SAM2 requires JPEG folder or MP4; PNG not supported for folder mode
            pngs = list(self.img_dir.glob("*.png"))
            if pngs:
                raise RuntimeError(f"{self.sample}: PNG frames detected; convert to JPG first.")
            raise RuntimeError(f"{self.sample}: no frames found under {self.img_dir}")

        # inst_id (str like '000') -> sorted list of annotated frame indices (int)
        self.inst_to_frames: Dict[str, List[int]] = {}
        for sub in sorted(p for p in self.ann_dir.iterdir() if p.is_dir()):
            frames = sorted(sub.glob("*.png"))
            if not frames:
                continue
            idxs = []
            for f in frames:
                stem = f.stem
                try:
                    idxs.append(int(stem))
                except ValueError:
                    continue
            if idxs:
                self.inst_to_frames[sub.name] = sorted(set(idxs))

        if not self.inst_to_frames:
            raise RuntimeError(f"{self.sample}: no annotated instances found under {self.ann_dir}")

        # For convenience
        self.H, self.W = np.array(Image.open(self.frame_paths[0]).convert("L")).shape

    def first_prompt(self) -> Tuple[int, str, np.ndarray]:
        best = None
        for inst_id, idxs in self.inst_to_frames.items():
            if not idxs:
                continue
            t0 = idxs[0]
            mp = self.ann_dir / inst_id / f"{t0:05d}.png"
            m = np.array(Image.open(mp))
            if m.ndim == 3:
                m = m[..., 0]
            m_bin = (m > 0).astype(np.uint8)
            if best is None or t0 < best[0]:
                best = (t0, inst_id, m_bin)
        if best is None:
            raise RuntimeError(f"{self.sample}: could not find any annotated frame to prompt")
        return best  # (frame_idx, inst_id, mask)


# =========================
# Video Samples encapsulated as DataLoader
# =========================

class SAVVideoDataset:
    def __init__(self, sav_root: Path, list_file: Path):
        self.sav_root = sav_root
        self.sample_names: List[str] = read_list(list_file)

    def __len__(self) -> int:
        return len(self.sample_names)

    def __getitem__(self, idx: int) -> SAVJPEGIndex:
        sample = self.sample_names[idx]
        return SAVJPEGIndex(sample, self.sav_root)

    def get_sample_name(self, idx: int) -> str:
        return self.sample_names[idx]


# =========================
# Model Construction and Forward Interface
# =========================

def build_sam2_predictor(model_cfg: str | Path, checkpoint: str | Path, device: str = "cuda"):
    return build_sam2_video_predictor(str(model_cfg), str(checkpoint), device=device)


def propagate_video(
    predictor,
    video_index: SAVJPEGIndex,
    prompt_idx: int,
    prompt_mask: np.ndarray,
    extra_prompts: List[Tuple[int, int, np.ndarray]] | None = None,
) -> Dict[int, Dict[int, np.ndarray]]:

    with torch.inference_mode():
        state = predictor.init_state(str(video_index.img_dir),
                                     offload_video_to_cpu=True,
                                     offload_state_to_cpu=True)
        # Primary prompt as obj_id=1
        predictor.add_new_mask(state, frame_idx=prompt_idx, obj_id=1, mask=(prompt_mask > 0))

        # Add extra prompts if any
        if extra_prompts:
            for obj_id, t_idx, m in extra_prompts:
                try:
                    predictor.add_new_mask(state, frame_idx=t_idx, obj_id=obj_id, mask=(m > 0))
                except Exception:
                    # Fallback to box (rare)
                    ys, xs = np.where(m > 0)
                    if len(xs) == 0:
                        continue
                    box = np.array([[xs.min(), ys.min(), xs.max(), ys.max()]], dtype=np.float32)
                    predictor.add_new_points_or_box(state, frame_idx=t_idx, obj_id=obj_id, box=box)

        out: Dict[int, Dict[int, np.ndarray]] = {}
        for fidx, obj_ids, masks in predictor.propagate_in_video(state):
            per_obj: Dict[int, np.ndarray] = {}
            for oid, m in zip(obj_ids, masks):
                # convert to 2D binary numpy
                if torch is not None and isinstance(m, torch.Tensor):
                    m_np = m.detach().to("cpu").numpy()
                else:
                    m_np = np.array(m)
                m_np = np.squeeze(m_np)
                if m_np.ndim > 2:
                    axes = tuple(range(0, m_np.ndim - 2))
                    m_np = m_np.max(axis=axes)
                if m_np.ndim != 2:
                    m_np = np.reshape(m_np, m_np.shape[-2:])
                if np.issubdtype(m_np.dtype, np.floating):
                    m_bin = (m_np > 0.5).astype(np.uint8)
                elif m_np.dtype == bool:
                    m_bin = m_np.astype(np.uint8)
                else:
                    m_bin = (m_np > 0).astype(np.uint8)
                per_obj[int(oid)] = m_bin
            out[int(fidx)] = per_obj
        return out


def calibrate_video(
    predictor,
    video_index: SAVJPEGIndex,
    prompt_idx: int,
    prompt_mask: np.ndarray,
    extra_prompts: List[Tuple[int, int, np.ndarray]] | None = None,
    max_frames: int | None = None,
    cali_type: str | None = None
) -> None:

    with torch.inference_mode():
        state = predictor.init_state(str(video_index.img_dir),
                                     offload_video_to_cpu=True,
                                     offload_state_to_cpu=True)
        # Primary prompt
        predictor.add_new_mask(
            state,
            frame_idx=prompt_idx,
            obj_id=1,
            mask=(prompt_mask > 0),
        )

        if extra_prompts:
            for obj_id, t_idx, m in extra_prompts:
                try:
                    predictor.add_new_mask(
                        state,
                        frame_idx=t_idx,
                        obj_id=obj_id,
                        mask=(m > 0),
                    )
                except Exception:
                    ys, xs = np.where(m > 0)
                    if len(xs) == 0:
                        continue
                    box = np.array(
                        [[xs.min(), ys.min(), xs.max(), ys.max()]],
                        dtype=np.float32,
                    )
                    predictor.add_new_points_or_box(
                        state,
                        frame_idx=t_idx,
                        obj_id=obj_id,
                        box=box,
                    )

        for i, (fidx, obj_ids, masks) in enumerate(predictor.propagate_in_video(state)):
            if i == 0 and cali_type is not None:
                enable_calibration_woquantization(predictor, quantizer_type=cali_type)
            if max_frames is not None and (i + 1) >= max_frames:
                break


# =========================
# Single Sample Prediction & Calibration
# =========================

def predict_sample(
    index: SAVJPEGIndex,
    predictor,
    pred_root: Path | None,
    save_preds: bool,
    skip_first_last: bool,
) -> Tuple[List[float], List[float]]:

    inst_ids = sorted(index.inst_to_frames.keys())
    inst_to_oid: Dict[str, int] = {inst: (i + 1) for i, inst in enumerate(inst_ids)}

    # Primary prompt
    t0, inst0, m0 = index.first_prompt()

    # Extra prompts
    extra: List[Tuple[int, int, np.ndarray]] = []
    for inst in inst_ids:
        if inst == inst0:
            continue
        frames = index.inst_to_frames[inst]
        if not frames:
            continue
        t_k = frames[0]
        mp = index.ann_dir / inst / f"{t_k:05d}.png"
        m = np.array(Image.open(mp))
        if m.ndim == 3:
            m = m[..., 0]
        m_bin = (m > 0).astype(np.uint8)
        extra.append((inst_to_oid[inst], t_k, m_bin))

    # Run propagation
    out = propagate_video(
        predictor=predictor,
        video_index=index,
        prompt_idx=t0,
        prompt_mask=m0,
        extra_prompts=[(inst_to_oid[inst0], t0, m0)] + extra,
    )

    J_list: List[float] = []
    F_list: List[float] = []

    # Save predictions ONLY for GT frames, per instance folder; and compute metrics online
    for inst in inst_ids:
        oid = inst_to_oid[inst]
        ann_frames = index.inst_to_frames[inst]
        # SA-V protocol: skip first and last annotated frames by default
        frs = ann_frames
        if skip_first_last and len(frs) >= 3:
            frs = frs[1:-1]
        for f in frs:
            m_bin = out.get(f, {}).get(oid, None)
            if m_bin is None:
                m_bin = np.zeros((index.H, index.W), dtype=np.uint8)
            # metrics
            gt = np.array(Image.open(index.ann_dir / inst / f"{f:05d}.png"))
            if gt.ndim == 3:
                gt = gt[..., 0]
            gt_bin = (gt > 0).astype(np.uint8)
            J_list.append(iou(m_bin, gt_bin))
            F_list.append(f_measure(m_bin, gt_bin))
            # optional saving
            if save_preds and pred_root is not None:
                save_mask(pred_root / index.sample / inst / f"{f:05d}.png", m_bin)

    return J_list, F_list


def calibrate_sample(
    index: SAVJPEGIndex,
    predictor,
    max_frames: int | None = None,
    cali_type: str | None = None
) -> None:

    inst_ids = sorted(index.inst_to_frames.keys())
    inst_to_oid: Dict[str, int] = {inst: (i + 1) for i, inst in enumerate(inst_ids)}

    # primary prompt
    t0, inst0, m0 = index.first_prompt()

    # extra prompts
    extra: List[Tuple[int, int, np.ndarray]] = []
    for inst in inst_ids:
        if inst == inst0:
            continue
        frames = index.inst_to_frames[inst]
        if not frames:
            continue
        t_k = frames[0]
        mp = index.ann_dir / inst / f"{t_k:05d}.png"
        m = np.array(Image.open(mp))
        if m.ndim == 3:
            m = m[..., 0]
        m_bin = (m > 0).astype(np.uint8)
        extra.append((inst_to_oid[inst], t_k, m_bin))

    calibrate_video(
        predictor=predictor,
        video_index=index,
        prompt_idx=t0,
        prompt_mask=m0,
        extra_prompts=[(inst_to_oid[inst0], t0, m0)] + extra,
        max_frames=max_frames,
        cali_type=cali_type
    )


def calibrate(
    dataset: SAVVideoDataset,
    predictor,
    num_sample: int,
    num_frame: int | None,
    shuffle: bool = True,
    cali_type: str | None = None
):

    indices = list(range(len(dataset)))
    if shuffle:
        random.shuffle(indices)
    indices = indices[:num_sample]

    print(f"=== Calibration with {len(indices)} videos ===")
    for i, idx in enumerate(indices, 1):
        index = dataset[idx]
        print(f"[Calib {i}/{len(indices)}] sample {index.sample}")
        calibrate_sample(
            index=index,
            predictor=predictor,
            max_frames=num_frame,
            cali_type=cali_type
        )


# =========================
# Quantization
# =========================

def quantize_model(model, config_quant):

    def replace_module(module, w_qconfig, a_qconfig, ahcqsam_config, ptq4sam_config):
        for name, child in module.named_children():
            if type(child) in specials:
                setattr(module, name, specials[type(child)](child, w_qconfig, a_qconfig, ahcqsam_config, ptq4sam_config))
            else:
                replace_module(child, w_qconfig, a_qconfig, ahcqsam_config, ptq4sam_config)

    replace_module(model, config_quant.w_qconfig, config_quant.a_qconfig, config_quant.ahcqsam, config_quant.ptq4sam)
    return model


def recon_model(predictor, fp_predictor, cali_data, q_config):

    def _recon_model(module, fp_module):
        for name, child_module in module.named_children():
            if isinstance(child_module, (QuantizedLayer, QuantizedBlock, PreQuantizedLayer, QuantizedMatMul)):
                print('Start reconstruction for module:\n{}'.format(str(child_module)))
                reconstruction(predictor, fp_predictor, child_module, getattr(fp_module, name), cali_data, q_config)
            else:
                _recon_model(child_module, getattr(fp_module, name))

    # Start reconstruction
    _recon_model(predictor, fp_predictor)


def condiBasedAct_recon_model(model, fp_model, cali_data, q_config):
    enable_quantization(model, 'act_fake_quantize')

    def _condiBasedAct_recon_model(module, fp_module, prefix=""):
        for name, child_module in module.named_children():
            global_name = f"{prefix}.{name}" if prefix else name

            if isinstance(child_module, (QuantizedLayer, PreQuantizedLayer)):

                print('Begin ACNR Reconstruction for Module [{}]:\n'.
                      format(global_name))

                cond_num_before_optimization, cond_num_after_optimization = \
                    condiBasedAct_reconstruction(model, fp_model, child_module,
                                                 getattr(fp_module, name), cali_data, q_config)

                if cond_num_before_optimization < cond_num_after_optimization:
                    print('#####')
                    print('#####')
                    print('Module [{}], Cond_Num_Before_Optimization: {}, ' \
                          'Cond_Num_After_Optimization: {}'.
                          format(global_name, cond_num_before_optimization,
                                 cond_num_after_optimization))
                    print('#####')
                    print('#####')

            else:
                _condiBasedAct_recon_model(child_module, getattr(fp_module, name), global_name)

    _condiBasedAct_recon_model(model, fp_model, prefix="")


# =========================
# Main Quantization Function
# =========================

def main():
    ap = argparse.ArgumentParser(description="Predict SAM2 on SA-V Test Pack")
    ap.add_argument("--sav-test", default='/home/dataset/SA-V/sav_test', type=Path, help="Root of SA-V pack for test (contains JPEGImages_24fps & Annotations_6fps)")
    ap.add_argument("--sav-train", default='/home/dataset/SA-V/sav_train', type=Path, help="Root of SA-V pack for training")
    ap.add_argument("--model-cfg", default='./configs/sam2.1/sam2.1_hiera_t.yaml', type=Path, help="Path to SAM2 model configs")
    ap.add_argument("--checkpoint", default='/home/zhang/Project/sam2/sam2/checkpoints/sam2.1_hiera_tiny.pt', type=Path, help="Path to SAM2 model checkpoint")
    ap.add_argument("--pred-root", default=None, help="Where to save predictions. Omit with None")
    ap.add_argument('--fp-model', action='store_true', default=False, help="Run with FP model")
    ap.add_argument('--q_config', type=str, default='./exp/config44.yaml', help="Quantization configuration")
    ap.add_argument('--recon', action='store_true', default=False, help="Optimize model by reconstruction")
    ap.add_argument("--device", default="cuda", type=str)
    args = ap.parse_args()

    if args.q_config:
        q_config = utils.parse_config(args.q_config)

    save_preds = args.pred_root is not None
    args.pred_root = Path(args.pred_root) if args.pred_root else None

    # ========= Create Dataset =========
    train_suffix = args.sav_train.name
    train_list_file = args.sav_train / f"{train_suffix}.txt"
    train_dataset = SAVVideoDataset(args.sav_train, train_list_file)

    test_suffix = args.sav_test.name
    test_list_file = args.sav_test / f"{test_suffix}.txt"
    test_dataset = SAVVideoDataset(args.sav_test, test_list_file)

    # ========= Create Predictor =========
    predictor = build_sam2_predictor(
        model_cfg=args.model_cfg,
        checkpoint=args.checkpoint,
        device=args.device,
    )

    if not args.fp_model:

        # Quantize Model
        predictor = quantize_model(predictor, q_config)
        predictor.cuda()
        predictor.eval()

        # Copy as FP Counterpart
        fp_predictor = deepcopy(predictor)
        disable_all(fp_predictor)

        for name, module in predictor.named_modules():
            if isinstance(module, ObserverBase):
                module.set_name(name)

        # ========= Calibration =========
        num_sample = min(q_config.calibrate.sample, len(train_dataset))
        num_frame = q_config.calibrate.frame or None

        # 0) BIG
        if q_config.ptq4sam.BIG:
            calibrate(train_dataset, predictor, num_sample=1, num_frame=2, shuffle=True, cali_type=None)
            bimodal_adjust(predictor)
        # 0.5) Condition
        if q_config.ahcqsam.acnr:
            calibrate(train_dataset, predictor, num_sample, num_frame + 1, shuffle=True, cali_type='act_fake_quant')
            condiBasedAct_recon_model(predictor, fp_predictor, train_dataset, q_config)
        # 1) Calibrate Activations
        calibrate(train_dataset, predictor, num_sample, num_frame + 1, shuffle=True, cali_type='act_fake_quant')
        # 2) Calibrate Weights
        disable_all(predictor)
        calibrate(train_dataset, predictor, num_sample=1, num_frame=2, shuffle=True, cali_type='weight_fake_quant')

        # Activate FakeQuant
        enable_quantization(predictor)

        if args.recon:
            # Model Reconstruction
            print('Begin Model Reconstruction...')
            recon_model(predictor, fp_predictor, train_dataset, q_config)

            enable_quantization(predictor)

            for n, m in predictor.named_modules():
                if hasattr(m, 'drop_prob'):
                    m.drop_prob = 1

    # ========= Test and Evaluate Model =========
    all_J: List[float] = []
    all_F: List[float] = []

    for i in range(len(test_dataset)):
        index = test_dataset[i]
        print(f"[{i+1}/{len(test_dataset)}] Predicting {index.sample} ...")
        J_list, F_list = predict_sample(
            index=index,
            predictor=predictor,
            pred_root=args.pred_root,
            save_preds=save_preds,
            skip_first_last=True,
        )
        if J_list:
            print(
                f"  -> Sample {index.sample}: "
                f"J={np.mean(J_list)*100:5.2f}  "
                f"F={np.mean(F_list)*100:5.2f}  "
                f"J&F={(0.5*(np.mean(J_list)+np.mean(F_list)))*100:5.2f}"
            )
        all_J.extend(J_list)
        all_F.extend(F_list)

    # Evaluate over dataset
    mean_J = float(np.mean(all_J)) if all_J else 0.0
    mean_F = float(np.mean(all_F)) if all_F else 0.0
    mean_JF = 0.5 * (mean_J + mean_F)
    print("========== Online Evaluation (SA-V) ==========")
    print(f"Frames(eval): {len(all_J)}")
    print(f"Mean J  (IoU):   {mean_J*100:.2f}")
    print(f"Mean F  (Bound): {mean_F*100:.2f}")
    print(f"Mean J&F:        {mean_JF*100:.2f}")

    if save_preds:
        print("Predictions saved to: ", args.pred_root)


if __name__ == "__main__":
    main()
