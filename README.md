# AHCQ-SAM: Toward Accurate and Hardware-Compatible Post-Training Segment Anything Model Quantization

## 1. Overview

* `./sam`: PTQ framework for SAM.
* `./sam2`: PTQ framework for SAM2.

We use NVIDIA A100 GPU with 80G memory to perform the experiments. Please follow the instruction in each directory to run experiments, respectively.

## 2. Abstract

The Segment Anything Model (SAM) has revolutionized image and video segmentation with its powerful zero-shot capabilities. However, its massive parameter scale and high computational demands hinder efficient deployment on resource-constrained edge devices. While Post-Training Quantization (PTQ) offers a practical solution, existing methods still fail to handle four critical quantization challenges: (1) ill-conditioned weights; (2) skewed and long-tailed post-GELU activations; (3) pronounced inter-channel variance in linear projections; and (4) exponentially scaled and heterogeneous attention scores. To mitigate these bottlenecks, we propose AHCQ-SAM, an accurate and hardware-compatible PTQ framework featuring four synergistic components: (1) Activation-aware Condition Number Reduction (ACNR), which regularizes weight matrices via a proximal point algorithm to suppress ill-conditioning; (2) Hybrid Log-Uniform Quantization (HLUQ), which combines power-of-two and uniform quantizers to capture skewed post-GELU activations; (3) Channel-Aware Grouping (CAG), which clusters channels with homogeneous statistics to achieve high accuracy with minimal hardware overhead; and (4) Logarithmic Nonlinear Quantization (LNQ), which utilizes logarithmic transformations to adaptively adjust quantization resolution for exponential and heterogeneous attention scores. Experimental results demonstrate that AHCQ-SAM outperforms current methods on SAM. Compared with the SOTA method, it achieves a 15.2% improvement in mAP for 4-bit SAM-B with Faster R-CNN on the COCO dataset. Furthermore, we establish a PTQ benchmark for SAM2, where AHCQ-SAM yields a 14.01% improvement in J&F for 4-bit SAM2-Tiny on the SA-V Test dataset. Finally, FPGA-based implementation validates the practical utility of AHCQ-SAM, delivering a 7.12x speedup and a 6.62x power efficiency improvement over the floating-point baseline.

**Update on Aug. 22, 2026**

We release our code implementation of [PQ-SAM](https://eccv.ecva.net/virtual/2024/poster/2654) (ECCV'24), [Mix-QSAM](https://openaccess.thecvf.com/content/CVPR2025W/eLVM/html/Ranjan_Mix-QSAM_Mixed-Precision_Quantization_of_the_Segment_Anything_Model_CVPRW_2025_paper.html) (CVPRW'25), [Mix-QSAM2](https://ojs.aaai.org/index.php/AAAI/article/view/37374) (AAAI'26), and AHCQ-SAM on COCO image segmentation task for reference. Please follow the README guidance in `./baselines` and each sub-directory to run the experiments.

## Citation

If you find this repo is useful, please cite our paper. Thanks.

```bibtex
@article{zhang2025ahcq,
  title={AHCQ-SAM: Toward Accurate and Hardware-Compatible Post-Training Segment Anything Model Quantization},
  author={Zhang, Wenlun and Zhong, Yunshan and Yan, Weiqi and Zhang, Shengchuan and Ando, Shimpei and Yoshioka, Kentaro},
  journal={arXiv preprint arXiv:2503.03088},
  year={2025}
}
```

## Acknowledgments
Code is built upon our previous work [AHCPTQ](https://github.com/Keio-CSG/AHCPTQ).
