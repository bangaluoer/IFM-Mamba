# Isotropic Frequency-Modulated Mamba for High-Fidelity Structure-Preserving Image Denoising

[![Python](https://img.shields.io/badge/Python-3.8-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0.1-EE4C2C.svg)](https://pytorch.org/)

> **🚨 Official Statement:** > This code is directly related to the manuscript currently submitted to **The Visual Computer (TVC)**. If you find our code, pre-trained models, or methodology helpful in your research, we highly encourage you to cite our relevant manuscript.

## 📌 Checkpoints Availability (Defensive Open-Source)
To ensure the reproducibility of our core claims, this repository currently provides the **inference code** and the **pre-trained weights for noise level $\sigma=15$**, which directly corresponds to the main performance comparison (Figure 1 in the paper) evaluated on the structure-heavy **Urban100 dataset**. 

The full suite of pre-trained models for all noise levels (e.g., $\sigma=25, 50$) and the complete training pipeline (including customized data augmentation and loss functions) will be officially released upon the acceptance of the manuscript.

---

## ⚙️ Environment Setup

We recommend using Anaconda to manage the environment. The code has been tested with Python 3.8.18, PyTorch 2.0.1, and CUDA 11.7.

**Step 1: Create a virtual environment**
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
The quantitative results (PSNR/SSIM) reported in the paper were evaluated on a single NVIDIA RTX 3090 GPU with PyTorch 2.0.1 and CUDA 11.7. Due to non-deterministic behaviors in PyTorch's floating-point operations across different GPU architectures and varying cuDNN versions, you may observe minor numerical fluctuations (typically within ±0.05dB in PSNR) when running the inference code on your local machine. This is a normal phenomenon in deep learning and does not affect the overall performance conclusion or structural superiority. We have fixed all random seeds in test.py to minimize such variations.

## 📝 Citation
If you find this work useful, please consider citing:
```bibetex
@article{ji2026isotropic,
  title={Isotropic Frequency-Modulated Mamba for High-Fidelity Structure-Preserving Image Denoising},
  author={Ji, Mintao and Zhou, Ying and Yuan, Mingsi and Hu, Xiaopeng and Wang, Fan},
  journal={Submitted to The Visual Computer},
  year={2026}
}
```
(The citation will be updated once the paper is officially published.)
