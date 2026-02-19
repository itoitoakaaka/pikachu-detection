# pikachu-detection

Real-time Pikachu detection using YOLOv8 trained on a custom Roboflow dataset.

## Overview

This project trains a YOLOv8 object detection model to detect Pikachu characters in images. The dataset is sourced from [Roboflow Universe](https://universe.roboflow.com/gian-b0euc/pikachu-zxwjc/dataset/1) and contains labeled images of various Pikachu illustrations, toys, and screenshots. Both a Google Colab notebook (`pikachu.ipynb`) and a standalone Python script (`pikachu_train.py`) are provided.

## Features

- **Dataset Download**: Automatically downloads the Pikachu detection dataset from Roboflow.
- **Model Training**: Fine-tunes a pre-trained YOLOv8s model on the Pikachu dataset.
- **Inference**: Runs detection on new images and saves annotated results.

## Dataset

- **Source**: Roboflow (CC BY 4.0 license)
- **Classes**: 1 (Pikachu)
- **Split**: 64 train / 18 validation / 10 test images

## Requirements

- Python 3.8+
- ultralytics (YOLOv8)
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
# 1. Download the dataset (requires Roboflow API key)
python pikachu_train.py download --api_key YOUR_ROBOFLOW_API_KEY

# 2. Train the model
python pikachu_train.py train --data_dir Pikachu-1 --epochs 50

# 3. Run inference on an image
python pikachu_train.py predict --model runs/detect/train/weights/best.pt --image test.jpg
```

Alternatively, open `pikachu.ipynb` in Google Colab to run the training interactively with GPU support.

## Output

- `runs/detect/train/weights/best.pt`: Best model weights.
- `pikachu_detection_result.jpg`: Annotated detection result image.
