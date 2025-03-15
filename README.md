# 🍅 TomatoSorter AI 🤖

```
UPDATE
Removed the heavy load from the git. send an email (kasundularaam@gmail.com) to request for the dataset and videos.
```

> **Smart tomato classification system using computer vision and machine learning**

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c)](https://pytorch.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.5%2B-green)](https://opencv.org/)

## 🌟 Overview

TomatoSorter AI is a complete pipeline for automatically classifying tomatoes into three categories: ripe, unripe, and damaged. The system uses a combination of color analysis and deep learning to achieve accurate sorting, perfect for quality control in agricultural settings.

## 🔍 Features

- 🎥 **Video Frame Extraction**: Automatically extract frames from tomato videos
- 🧠 **Pre-trained Model**: Comes with a trained ResNet-50 model (transfer learning)
- 🎯 **Multi-class Classification**: Sorts tomatoes into three categories:
  - 🟢 Unripe (yellow-green tomatoes)
  - 🔴 Ripe (red, healthy tomatoes)
  - ⚠️ Damaged (tomatoes with physical damage)
- 📊 **Analysis Tools**: Includes visualization of training results
- 🔄 **Batch Processing**: Process multiple images at once

## 🛠️ Installation

1. **Clone the repository**
```bash
git clone https://github.com/kasundularaam/tomato-sorter-ai.git
cd tomato-sorter-ai
```

2. **Set up a virtual environment**
```bash
# Create a virtual environment
python -m venv venv

# Activate it
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

## 🚀 Usage

### Classifying tomatoes with the pre-trained model

```bash
python main.py
```
This will:
1. Load tomato images from the `tomatoes` directory
2. Classify them using color analysis (for unripe) and the pre-trained model (for ripe vs damaged)
3. Sort the images into `processed/ripe`, `processed/unripe`, and `processed/damaged` directories

### Creating your own dataset from videos

```bash
python prepare_dataset.py
```
This extracts frames from tomato videos organized in `videos/ripe`, `videos/unripe`, and `videos/damaged` directories.

### Training a new model

```bash
python train.py
```
This trains a new classification model on the extracted frames, saving the best model as `best_tomato_model.pth`.

### Testing a model

```bash
python try_model.py
```
This tests a trained model on new images, showing its raw classification capabilities without the color analysis for unripe tomatoes.

## 📁 Project Structure

```
tomato-sorter-ai/
├── main.py              # Main script for classifying tomatoes
├── prepare_dataset.py   # Script for extracting frames from videos
├── train.py             # Script for training the classification model
├── try_model.py         # Script for testing the raw model predictions
├── best_tomato_model.pth # Pre-trained model weights
├── requirements.txt     # Python dependencies
├── tomatoes/            # Directory for input tomato images
├── processed/           # Directory for sorted output images
├── dataset/             # Directory containing extracted frames
│   ├── ripe/
│   ├── unripe/
│   └── damaged/
├── videos/              # Directory containing tomato videos
│   ├── ripe/
│   ├── unripe/
│   └── damaged/
└── results/             # Directory containing training visualizations
```

## 📊 Model Performance

The included pre-trained model achieves:
- 95%+ accuracy on ripe vs damaged classification
- Highly accurate unripe detection using specialized color analysis

## 🔧 Configuration

You can adjust the system's behavior by modifying the constants at the top of each script:

- **main.py**: Change input/output directories, confidence threshold
- **prepare_dataset.py**: Modify frame extraction rate, resize dimensions
- **train.py**: Adjust training parameters like epochs, batch size, learning rate

## 🤝 Contributing

Contributions are welcome! Feel free to open issues or submit pull requests.

## 🙏 Acknowledgements

- PyTorch team for the amazing deep learning framework
- OpenCV community for computer vision tools
- All contributors to this project

---

Happy tomato sorting! 🍅✨