# PTQ Framework for SAM 2

## 1. Environment Settings

### 1.1 Create Environment

1. Install PyTorch
```
conda create -n ahcqsam_sam2 python=3.10 -y
conda activate ahcqsam_sam2
cd ./sam2
pip install torch==2.10.0 torchvision==0.25.0 torchaudio==2.10.0 --index-url https://download.pytorch.org/whl/cu128
```

2. Install SAM 2

```
cd ./sam2
pip install -e .
```

3. Download Checkpoints

```
cd ./checkpoints
./download_ckpts.sh
cd ../..
```

### 1.2 Prepare Dataset

Download the SA-V and DAVIS datasets. Re-collect the dataset using the scripts in ``./dataset``. The final SA-V dataset should be arranged as follows:

```
├── SA-V
│   ├── sav_train
│   │   ├── Annotations_6fps
│   │   ├── JPEGImages_24fps
│   │   ├── sav_train.txt
│   ├── sav_val
│   │   ├── Annotations_6fps
│   │   ├── JPEGImages_24fps
│   │   ├── sav_val.txt
│   ├── sav_test
│   │   ├── Annotations_6fps
│   │   ├── JPEGImages_24fps
│   │   ├── sav_test.txt
```

Similarly, DAVIS 2017 dataset will be arranged as the following format:

```
├── DAVIS_2017
│   ├── sav_train
│   │   ├── Annotations_6fps
│   │   ├── JPEGImages_24fps
│   │   ├── sav_train.txt
│   ├── sav_val
│   │   ├── Annotations_6fps
│   │   ├── JPEGImages_24fps
│   │   ├── sav_val.txt
```

## 2. Run Experiments

Please use the following command to perform AHCQ-SAM quantization:

```
python ahcqsam/solver/test_quant.py \
--sav-test <PATH_TO_SAV-TEST> \
--sav-train <PATH_TO_SAV-TRAIN> \
--model-cfg ./configs/sam2.1/<MODEL>.yaml \
--checkpoint ./sam2/checkpoints/<MODEL>.pt \
--q_config ./exp/<QCONFIG>.yaml \
--recon
```

Here, ``<PATH_TO_SAV-TEST>`` and ``<PATH_TO_SAV-TRAIN>`` are the path to test and training set of dataset (SA-V or DAVIS), ``<MODEL>`` is the corresponding SAM2 model, and ``<QCONFIG>.yaml`` is the specific quantization configuration file.

For example, to perform W4A4 quantization for SAM2.1-Tiny, use the following command:

```
python ahcqsam/solver/test_quant.py \
--sav-test <YOUR_SAV-TEST_ROOT> \
--sav-train <YOUR_SAV-TRAIN_ROOT> \
--model-cfg ./configs/sam2.1/sam2.1_hiera_t.yaml \
--checkpoint ./sam2/checkpoints/sam2.1_hiera_tiny.pt \
--q_config ./exp/config44.yaml \
--recon
```
