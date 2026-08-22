# Baseline entry points

Run these scripts with this directory as the working directory:

```bash
cd /path/to/AHCQ-SAM/baselines
```

- `mix_qsam/test_quant.py`: Mix-QSAM for SAM.
- `pq_sam/test_quant.py`: PQ-SAM for SAM.
- `mix_qsam2/test_quant_image.py`: Mix-QSAM2 image evaluation.
- `mix_qsam2/test_quant_video.py`: Mix-QSAM2 video evaluation.
- `sam2_image/test_quant.py`: AHCQ-SAM2 image evaluation baseline.

All entry points derive imports from the repository location. SAM detector configs and checkpoint paths are anchored to `../sam`; SAM2 sources and checkpoints are anchored to `../sam2`.
