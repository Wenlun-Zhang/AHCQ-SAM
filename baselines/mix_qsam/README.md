# Mix-QSAM Usage

This directory contains the Mix-QSAM reproduction entry for SAM instance
segmentation PTQ. It runs bit allocation, quantizer replacement, PTQ4SAM
calibration, and COCO segmentation evaluation in one command.

## Environment

Start from the existing SAM PTQ environment:

```bash
conda activate ahcqsam_sam
cd /path/to/AHCQ-SAM/baselines
```

Install CVXPY and a mixed-integer solver. CBC is recommended because ECOS_BB can
be slow on the fine-grained allocation problem.

```bash
conda install -c conda-forge cvxpy cylp coin-or-cbc
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
```

## Config

The default config is:

```bash
./mix_qsam/exp/config66.yaml
```

Important fields:

```yaml
mix_qsam:
    enabled: true
    solver: cvxpy
    cvxpy_solver: CBC
    use_bitops: true
    min_budget_ratio: 0.95
    candidate_bits: [4, 5, 6, 7, 8]
    target_bits: 6
    score_samples: 64
    max_boxes_per_image: 16
```

For a W4A4 run, use:

```yaml
candidate_bits: [2, 3, 4, 5, 6]
target_bits: 4
```

For a W5A5 run, use a lower search pool, for example:

```yaml
candidate_bits: [3, 4, 5, 6, 7]
target_bits: 5
```

## Run

End-to-end Mix-QSAM PTQ and COCO segmentation evaluation:

```bash
python ./mix_qsam/test_quant.py \
  --config ../sam/projects/configs/faster_rcnn/faster-rcnn-r50_sam-vit-b.py \
  --mix-config ./mix_qsam/exp/config66.yaml \
  --quant-encoder
```

Save the searched bit allocation JSON:

```bash
python ./mix_qsam/test_quant.py \
  --config ../sam/projects/configs/faster_rcnn/faster-rcnn-r50_sam-vit-b.py \
  --mix-config ./mix_qsam/exp/config66.yaml \
  --quant-encoder \
  --save-bit-allocation ./mix_qsam/bit_allocation.json
```

To reuse allocation scores between runs, set `mix_qsam.score_cache` in the selected
Mix-QSAM YAML file (for example, `./mix_qsam/exp/config66.yaml`). The cache stores importance, synergy, and MAC statistics; stale
caches are rejected when the layer-unit list does not match.
