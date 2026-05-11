# Frequency-Modulated Isotropic Mamba for Structure-Preserving Image Denoising

[![Python](https://img.shields.io/badge/Python-3.8-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0.1-EE4C2C.svg)](https://pytorch.org/)

## 📌 Overview

This repository provides the official implementation of **Frequency-Modulated Isotropic Mamba for Structure-Preserving Image Denoising**.

IFM-Mamba is a full-resolution isotropic image denoising framework designed to preserve fine geometric structures while suppressing noise. The model combines lightweight NAFBlocks for local texture extraction, Attentive State Space Models for long-range dependency modeling, and a lightweight learnable Frequency Gate for spectral modulation.

This repository currently provides inference code and pre-trained weights for evaluating the proposed method on standard image denoising benchmarks.

---

## 📌 Checkpoints Availability

To support reproducibility, this repository currently provides the **inference code** and the **pre-trained weights for Gaussian noise level $\sigma=15$**, which correspond to the main performance comparison evaluated on the structure-heavy **Urban100** dataset.

Additional pre-trained models for other noise levels, such as $\sigma=25$ and $\sigma=50$, as well as the complete training pipeline, will be released in a future update.

---

## ⚙️ Environment Setup

We recommend using Anaconda to manage the environment. The code has been tested with Python 3.8.18, PyTorch 2.0.1, and CUDA 11.7.

### Step 1: Create a virtual environment

```bash
conda create -n mamba_denoise python=3.8.18
conda activate mamba_denoise
```
**Step 2: Install PyTorch**
```bash
conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.7 -c pytorch -c nvidia
```
**Step 3: Install required dependencies You can quickly install all other dependencies using the provided requirements.txt**
```bash
pip install -r requirements.txt
```
(Note: Ensure that causal-conv1d and mamba-ssm are correctly compiled with your local CUDA version.)

## 📁 Directory Structure

Please organize the downloaded weights and datasets as follows before running the inference script:

```text
├── checkpoints/
│   └── best_model_sigma15.pth    # Download from the link below
├── datasets/
│   └── Urban100/                 # Place the Urban100 test images here
│       ├── image_001.png
│       ├── image_002.png
│       └── ...
├── models/
│   ├── mamba_ir.py               # Network architecture
│   └── ...
├── test.py                       # Main inference script
├── requirements.txt
└── README.md
```
## 🚀 Quick Start (Inference)
## 1. Download Pre-trained Weights
Download the pre-trained model for noise level σ=15 from the link below and place it in the ./checkpoints/ directory.

  Link: [Google Drive](https://drive.google.com/drive/folders/15_aAldDMwLL0GEX8K1RTFwSPyP35RD4N?usp=drive_link)

## 2. Run Evaluation
To evaluate the model on the Urban100 dataset (σ=15),simply run the following command:
```bash
python test.py --dataset_dir ./datasets/Urban100 --noise_level 15 --weights ./checkpoints/best_model_sigma15.pth
```
The denoised images will be saved in the ./results/ folder. You can then evaluate the PSNR/SSIM using your preferred metrics calculation script.

## 🛡️ Note on Reproducibility
The quantitative results reported in the manuscript were evaluated using PyTorch 2.0.1 and CUDA 11.7. Due to non-deterministic behaviors in PyTorch floating-point operations across different GPU architectures and cuDNN versions, minor numerical fluctuations may occur when running the inference code on different local environments.

In general, these fluctuations are expected to be small and do not affect the overall performance conclusion. Random seeds are fixed in test.py to improve reproducibility.

## 📝 Citation
If you find this work useful, please consider citing:
```bibetex
@article{ji2026frequency,
  title={Frequency-Modulated Isotropic Mamba for Structure-Preserving Image Denoising},
  author={Ji, Mintao and Zhou, Ying and Yuan, Mingsi and Hu, Xiaopeng and Wang, Fan},
  year={2026}
}
```
(The citation will be updated once the paper is officially published.)
## 📬 Contact

For questions about the code or models, please contact the authors through the GitHub issue page.
