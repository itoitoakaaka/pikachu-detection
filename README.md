# pikachu-detection

A compact computer-vision training project using YOLOv8 and a custom object-detection dataset.

## Project history

The original notebook / training exercise was created during my earlier Python and machine-learning training period (2023–2024). The project was subsequently reorganized and published on GitHub, so the repository history reflects later cleanup and maintenance rather than the original learning period.

## Overview

This project fine-tunes a pretrained YOLOv8 model to detect Pikachu instances in images. It includes both a Google Colab notebook and a standalone Python training script.

## What this project demonstrates

- Python-based machine-learning workflow
- object detection with YOLOv8
- dataset acquisition and train / validation / test handling
- model fine-tuning and inference
- transition from notebook experimentation to a reusable Python script

## Dataset

- Source: Roboflow Universe
- License: CC BY 4.0
- Classes: 1
- Split: 64 train / 18 validation / 10 test images

## Requirements

- Python 3.8+
- ultralytics
- roboflow
- OpenCV

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

```bash
python pikachu_train.py download --api_key YOUR_ROBOFLOW_API_KEY
python pikachu_train.py train --data_dir Pikachu-1 --epochs 50
python pikachu_train.py predict --model runs/detect/train/weights/best.pt --image test.jpg
```

The included notebook can also be run in Google Colab.

## Portfolio note

This is an early computer-vision learning project rather than a research result. It is kept public to show the progression from basic object detection toward my more recent work in human movement analysis, multimodal sensing, and human-centered AI.
