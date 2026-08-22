# PQ-SAM Usage

This directory contains the PQ-SAM baseline for detector-guided SAM instance-segmentation quantization and COCO evaluation.

## Environment

Use the SAM PTQ environment and run all commands from the repository's `baselines` directory:

```bash
conda activate ahcqsam_sam
cd /path/to/AHCQ-SAM/baselines
```

The SAM environment, MMDetection installation, compiled CUDA operators, COCO dataset, detector weights, and SAM checkpoints should be prepared as described in `../../sam/README.md`.

## Configurations

The available quantization configurations are:

```text
./pq_sam/exp/config44.yaml  # W4A4
./pq_sam/exp/config55.yaml  # W5A5
./pq_sam/exp/config66.yaml  # W6A6 (default)
```

Important `pq_sam` fields include:

- `gadt`: enable the PQ-SAM activation quantizer.
- `ohc`: initialize outlier-aware channel grouping from calibration data.
- `epochs`: number of quantization-parameter optimization epochs.
- `teacher_device`: FP teacher device used with `--dual-gpu-train`.
- `train_encoder`: include encoder quantization parameters in optimization.

## Run

W6A6 PQ-SAM with a YOLOX prompt detector:

```bash
python ./pq_sam/test_quant.py \
  --config ../sam/projects/configs/yolox/yolo_l-sam-vit-b.py \
  --q_config ./pq_sam/exp/config66.yaml \
  --quant-encoder
```

W4A4 with Faster R-CNN:

```bash
python ./pq_sam/test_quant.py \
  --config ../sam/projects/configs/faster_rcnn/faster-rcnn-r50_sam-vit-b.py \
  --q_config ./pq_sam/exp/config44.yaml \
  --quant-encoder
```

To place the quantized student on `--gpu-id` and the FP teacher on the configured `pq_sam.teacher_device`, add:

```bash
--dual-gpu-train --gpu-id 0
```

Without `--dual-gpu-train`, both models use `--gpu-id`; if the configured teacher GPU is unavailable, the code also falls back to `--gpu-id`.

Floating-point evaluation uses the same detector configuration:

```bash
python ./pq_sam/test_quant.py \
  --config ../sam/projects/configs/yolox/yolo_l-sam-vit-b.py \
  --fp-model
```

The detector configuration controls COCO paths, detector weights, and SAM checkpoint selection. Relative detector and checkpoint paths are automatically anchored to `../sam` when launched from `baselines`.