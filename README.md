# HAIS-SegFormer: A Lightweight Underwater Crack Segmentation Network

This repository contains the official PyTorch implementation of the paper:
**"HAIS-SegFormer: A Lightweight Underwater Crack Segmentation Network Based on Hybrid Attention and Feature Inhibition"** *(Under Review in JMSE)*.

## 📝 Abstract
Underwater crack detection is essential for the structural health monitoring of concrete dams. This study proposes **HAIS-SegFormer**, a lightweight network based on a Mix Transformer backbone. It introduces a tandem Hybrid Attention mechanism (CoordAtt + CBAM) and a bio-inspired Feature Inhibition Module (FIM) to actively suppress high-frequency noise such as water plants. Our model achieves a superior balance between segmentation accuracy (71.66% mIoU) and inference efficiency (73 FPS, 3.80M parameters).

## 🚀 Architecture


## 🛠️ Requirements
- Python 3.8+
- PyTorch 1.12.1+
- CUDA 11.3+
- torchvision, numpy, opencv-python

## 📊 Dataset
The public Underwater Crack Detection Dataset used in this study is available at Roboflow Universe:
[Download Dataset Here](https://universe.roboflow.com/arjun-elh2g/underwater-crack-detection-owcch)

## 💻 Usage

### 1. Train
To train the HAIS-SegFormer model on your custom dataset or the public underwater dataset, run:
```bash
python train.py --data_path ./dataset --epochs 100 --batch_size 8 --lr 1e-4
