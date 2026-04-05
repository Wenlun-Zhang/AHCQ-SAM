# PTQ Framework for SAM

## 1. Environment Settings

### 1.1 Create Environment

1. Install PyTorch
```
conda create -n ahcqsam_sam python=3.10 -y
conda activate ahcqsam_sam
pip install torch==2.0.0 torchvision==0.15.1 torchaudio==2.0.1 --index-url https://download.pytorch.org/whl/cu118
```

2. Install MMCV

```
pip install -U openmim
mim install "mmcv-full<2.0.0"
```

3. Install other requirements

```
cd ./sam
pip install -r requirements.txt
```

4. Compile CUDA operators

```
cd projects/instance_segment_anything/ops
python setup.py build install
cd ../../..
```

5. Install mmdet
```
cd mmdetection/
python3 setup.py build develop
cd ..
```

### 1.2 Prepare Dataset
Download the [COCO](https://drive.google.com/file/d/1j92XnlzQZwPff2sP_nwU3LE9Npemkn7Q/view?usp=sharing) dataset, recollect them as the following form, and revise the corresponding root directory in the code:

```
├── data
│   ├── coco
│   │   ├── annotations
│   │   ├── train2017
│   │   ├── val2017
│   │   ├── test2017
```
### 1.3 Download Model Weights

Download the model weights of SAM and detector, save them at the ``ckpt/`` folder:

| Model       | Download                                                                                    |
|-------------|---------------------------------------------------------------------------------------------|
| SAM-B       | [Link](https://drive.google.com/file/d/1UlwYWVRsS4SbSPDXlR5_dVmcuqT8CzeI/view?usp=sharing)  |
| SAM-L       | [Link](https://drive.google.com/file/d/14MBHh7OFwY8EpaGkX6ZyjUAw83wywk7U/view?usp=sharing)  |
| SAM-H       | [Link](https://drive.google.com/file/d/1fMJyX938_H17OxfVq6PQZ_ef9TBy5r36/view?usp=sharing)  |
| Faster-RCNN | [Link](https://drive.google.com/file/d/1RKTLk07E4apoRzwoeQbnaY8ZxEX1SlbG/view?usp=sharing)  |
| YOLOX       | [Link](https://drive.google.com/file/d/1FQeKOaDJzwqXq4zz8-VHJbn6iKFT4HLt/view?usp=sharing)  |
| HDETR       | [Link](https://drive.google.com/file/d/1i7iMAicmoif8tUbuHEntVtmEsJrpXTZ4/view?usp=sharing)  |
| DINO        | [Link](https://drive.google.com/file/d/1DDHkZcVI9TwmN9vqEYXFBjRZVsBK4yLO/view?usp=sharing)  |

## 2. Run Experiments

Please use the following command to perform AHCQ-SAM quantization:

```
python ahcqsam/solver/test_quant.py \
--config ./projects/configs/<DETECTOR>/<MODEL.py> \
--q_config ./exp/<QCONFIG>.yaml \
--quant-encoder
```

Here, ``<DETECTOR>`` is the folder name of prompt detector, ``<MODEL.py>`` is configuration file of corresponding SAM model, and ``<QCONFIG>.yaml`` is the specific quantization configuration file.

For example, to perform W4A4 quantization for SAM-B with a YOLO detector, use the following command:

```
python ahcqsam/solver/test_quant.py \
--config ./projects/configs/yolox/yolo_l-sam-vit-b.py \
--q_config ./exp/config44.yaml \
--quant-encoder
```
