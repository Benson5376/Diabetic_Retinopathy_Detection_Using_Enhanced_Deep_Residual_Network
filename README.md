# Diabetic Retinopathy Detection Using Enhanced Deep Residual Network

This project leverages deep learning techniques to classify diabetic retinopathy grades from retina images. It employs ResNet architectures for classification and evaluates performance through confusion matrices and accuracy visualizations.

![ResNet Architecture](https://github.com/Benson5376/Diabetic_Retinopathy_Detection_Using_Enhanced_Deep_Residual_Network/blob/main/resources/resnet.PNG)

## Table of Contents
- [Introduction](#introduction)
- [Features](#features)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Model Architectures](#model-architectures)
- [Evaluation Metrics](#evaluation-metrics)
- [Examples](#examples)
- [License](#license)

## Introduction
Diabetic retinopathy is a leading cause of blindness. Early detection through automated systems can assist in timely intervention. This project classifies retina images into one of five severity levels:
- 0: No DR
- 1: Mild
- 2: Moderate
- 3: Severe
- 4: Proliferative DR

The project consists of:
1. Custom DataLoader implementation.
2. ResNet18 and ResNet50 architectures (with and without pretraining).
3. Accuracy visualization and confusion matrix generation.

## Features
- Custom PyTorch DataLoader for flexible image preprocessing.
- Support for ResNet18 and ResNet50 models.
- Comparison between pretrained and non-pretrained models.
- Comprehensive evaluation using accuracy trends and confusion matrices.
- Adjustable hyperparameters for experimentation.

## Dataset
The dataset contains high-resolution retina images from the Kaggle Diabetic Retinopathy Detection competition:
- **Training set**: 28,100 images.
- **Testing set**: 7,025 images.
- Images are resized to 512x512 during preprocessing.

Classes:
- 0: No DR
- 1: Mild
- 2: Moderate
- 3: Severe
- 4: Proliferative DR

[Dataset source](https://www.kaggle.com/c/diabetic-retinopathy-detection)

## Installation
1. Clone the repository:
    ```bash
    git clone https://github.com/your-repo/diabetic-retinopathy-detection.git
    ```
2. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
3. Ensure PyTorch is installed for your system and GPU configuration. See [PyTorch Installation Guide](https://pytorch.org/get-started/locally/).

## Usage
### Training
Run the training script:
```bash
python Train.py
```
### Evaluation
Evaluate the trained model and visualize results:
```bash
python Evaluate.py
```

### Data Preprocessing
The `dataloader.py` script preprocesses images using:
- Resizing (512x512)
- Color jitter
- Random horizontal/vertical flips

## Model Architectures
This project implements:
- **ResNet18** and **ResNet50**
- Pretrained weights from ImageNet.
- Custom modifications to output a 5-class classification.

The models are defined in `Train.py` and are trained for:
- ResNet18: 10 epochs.
- ResNet50: 5 epochs.

### Hyperparameters
- Batch size: 4
- Learning rate: 1e-3
- Optimizer: SGD (Momentum: 0.9, Weight Decay: 5e-4)
- Loss Function: CrossEntropyLoss

## Evaluation Metrics
1. **Accuracy**: Compare the accuracy for pretrained vs. non-pretrained models across training and testing phases.
2. **Confusion Matrix**: Visualize misclassifications for detailed insights.


### Confusion Matrix
Generated using the `scikit-learn` library to summarize classification performance.

## Examples
1. **Model Training**:
    ```python
    python Train.py
    ```
2. **Accuracy Trend Plot**: Output graphs are saved as `.png` files.
3. **Confusion Matrix**:
    Example visualization for testing results.

## License
This project is licensed under the MIT License.


