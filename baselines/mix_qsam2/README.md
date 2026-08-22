# Mix-QSAM2 Usage

This directory contains Mix-QSAM2 mixed-precision quantization entry points for SAM2 image and video segmentation. It searches per-layer bit allocation, calibrates the quantized model, optionally performs reconstruction, and evaluates the resulting predictor.

## Environment

Use the SAM2 environment and run all commands from the repository's `baselines` directory:

```bash
conda activate ahcqsam_sam2
cd /path/to/AHCQ-SAM/baselines
```

Install the SAM2 package and checkpoints as described in `../../sam2/README.md`. Mixed-precision allocation may additionally require the optimization dependencies used by Mix-QSAM.

## Configurations

Image configurations:

```text
./mix_qsam2/exp_image/config44.yaml  # target W4A4
./mix_qsam2/exp_image/config55.yaml  # target W5A5
./mix_qsam2/exp_image/config66.yaml  # target W6A6
```

Video configurations:

```text
./mix_qsam2/exp_video/config44.yaml  # target W4A4
./mix_qsam2/exp_video/config55.yaml  # target W5A5
./mix_qsam2/exp_video/config66.yaml  # target W6A6
```

The `mix_qsam2` section controls candidate bits, target average bit-width, allocation-score samples, optional score caching, and SIS settings for video inference.

## COCO image evaluation

The COCO root must contain `train2017`, `val2017`, and `annotations`. Ground-truth boxes are used as SAM2 prompts.

```bash
python ./mix_qsam2/test_quant_image.py \
  --coco-root /path/to/coco \
  --model-cfg configs/sam2.1/sam2.1_hiera_t.yaml \
  --checkpoint ../sam2/sam2/checkpoints/sam2.1_hiera_tiny.pt \
  --q_config ./mix_qsam2/exp_image/config44.yaml \
  --save-bit-allocation ./mix_qsam2/image_bit_allocation.json \
  --save-json ./mix_qsam2/image_results.json
```

Add `--recon` to perform module reconstruction after calibration. For the floating-point image baseline, add `--fp-model`; the quantization configuration is then not used.

## SA-V video evaluation

Evaluate a training-calibrated model on SA-V Test:

```bash
python ./mix_qsam2/test_quant_video.py \
  --sav-train /path/to/SA-V/sav_train \
  --sav-test /path/to/SA-V/sav_test \
  --model-cfg configs/sam2.1/sam2.1_hiera_t.yaml \
  --checkpoint ../sam2/sam2/checkpoints/sam2.1_hiera_tiny.pt \
  --q_config ./mix_qsam2/exp_video/config44.yaml \
  --save-bit-allocation ./mix_qsam2/video_bit_allocation.json \
  --save-json ./mix_qsam2/video_results.json
```

Video evaluation reports region similarity J, boundary accuracy F, and their mean J&F. `--pred-root` optionally saves predicted masks. Add `--fp-model` to either video command for floating-point evaluation.

The SAM2 model configuration such as `configs/sam2.1/sam2.1_hiera_t.yaml` is a Hydra package configuration name, not a path relative to `baselines`.