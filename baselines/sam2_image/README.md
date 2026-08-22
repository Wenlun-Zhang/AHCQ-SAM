# SAM2 Image Quantization Experiments

This folder contains the COCO image-segmentation evaluation path for SAM2.
It supports the floating-point baseline and post-training quantization with
observer calibration and optional module reconstruction.

Floating-point example:

```bash
conda activate ahcqsam_sam2
cd /path/to/AHCQ-SAM/baselines

python sam2_image/test_quant.py \
  --fp-model \
  --coco-root /path/to/coco \
  --model-cfg configs/sam2.1/sam2.1_hiera_t.yaml \
  --checkpoint ../sam2/sam2/checkpoints/sam2.1_hiera_tiny.pt
```

Quantized example:

```bash
conda activate ahcqsam_sam2
cd /path/to/AHCQ-SAM/baselines

python ./sam2_image/test_quant.py \
  --coco-root /path/to/coco \
  --model-cfg configs/sam2.1/sam2.1_hiera_t.yaml \
  --checkpoint ../sam2/sam2/checkpoints/sam2.1_hiera_tiny.pt \
  --q_config ./sam2_image/exp/config66.yaml
```

Quantized example with reconstruction:

```bash
conda activate ahcqsam_sam2
cd /path/to/AHCQ-SAM/baselines

python sam2_image/test_quant.py \
  --coco-root /path/to/coco \
  --model-cfg configs/sam2.1/sam2.1_hiera_t.yaml \
  --checkpoint ../sam2/sam2/checkpoints/sam2.1_hiera_tiny.pt \
  --q_config ./sam2_image/exp/config66.yaml \
  --recon
```

The COCO root is expected to contain `train2017`, `val2017`, and
`annotations`. Quantized runs calibrate on `train2017` by default and evaluate
on `val2017`. The number of calibration images comes from
`q_config.calibrate.sample`; evaluation always runs the full validation split.
The reconstruction path uses `recon.sample` as the maximum number of collected
module IO samples and `recon.batch_size` as the number of samples optimized per
iteration.
