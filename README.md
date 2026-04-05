# AHCQ-SAM: Toward Accurate and Hardware-Compatible Post-Training Segment Anything Model Quantization

## 1. Overview

* `./sam`: PTQ framework for SAM.
* `./sam2`: PTQ framework for SAM2.

We use NVIDIA A100 GPU with 80G memory to perform the experiments. Please follow the instruction in each directory to run experiments, respectively.

## 2. Abstract

The Segment Anything Model (SAM) has revolutionized image and video segmentation with its powerful zero-shot capabilities. However, its massive parameter scale and high computational demands hinder efficient deployment on resource-constrained edge devices. While Post-Training Quantization (PTQ) offers a practical solution, existing methods still fail to handle four critical quantization challenges: (1) ill-conditioned weights; (2) skewed and long-tailed post-GELU activations; (3) pronounced inter-channel variance in linear projections; and (4) exponentially scaled and heterogeneous attention scores. To mitigate these bottlenecks, we propose AHCQ-SAM, an accurate and hardware-compatible PTQ framework featuring four synergistic components: (1) Activation-aware Condition Number Reduction (ACNR), which regularizes weight matrices via a proximal point algorithm to suppress ill-conditioning; (2) Hybrid Log-Uniform Quantization (HLUQ), which combines power-of-two and uniform quantizers to capture skewed post-GELU activations; (3) Channel-Aware Grouping (CAG), which clusters channels with homogeneous statistics to achieve high accuracy with minimal hardware overhead; and (4) Logarithmic Nonlinear Quantization (LNQ), which utilizes logarithmic transformations to adaptively adjust quantization resolution for exponential and heterogeneous attention scores. Experimental results demonstrate that AHCQ-SAM outperforms current methods on SAM. Compared with the SOTA method, it achieves a 15.2% improvement in mAP for 4-bit SAM-B with Faster R-CNN on the COCO dataset. Furthermore, we establish a PTQ benchmark for SAM2, where AHCQ-SAM yields a 14.01% improvement in $\mathcal{J}\&\mathcal{F}$ for 4-bit SAM2-Tiny on the SA-V Test dataset. Finally, FPGA-based implementation validates the practical utility of AHCQ-SAM, delivering a 7.12x speedup and a 6.62x power efficiency improvement over the floating-point baseline.

## Acknowledgments
Code is built upon our previous work [AHCPTQ](https://github.com/Keio-CSG/AHCPTQ).
