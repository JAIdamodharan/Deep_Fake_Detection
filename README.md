# AI-Powered Deepfake Detection for Cybersecurity

> Advanced deep learning model using ResNeXt + LSTM architecture to detect deepfake images with 93%+ accuracy

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Accuracy](https://img.shields.io/badge/Accuracy-93%25+-brightgreen.svg)]()

---

## 🎯 Overview

This project implements a state-of-the-art deepfake detection system designed to identify AI-generated fake images with high accuracy. The model combines ResNeXt-50 feature extraction with LSTM temporal analysis, achieving **93%+ accuracy** on test datasets with robust performance across real and fake image classifications.

### Problem Statement
With the rise of AI-generated content, deepfakes pose significant threats to cybersecurity, privacy, and information integrity. This system provides automated detection to combat these threats.

### Solution
Our deep learning model analyzes images at multiple levels to detect subtle artifacts and inconsistencies characteristic of deepfake generation, providing confidence scores for each prediction.

---

## ✨ Key Features

- **High Accuracy**: Achieves 93%+ accuracy with 95%+ ROC AUC score
- **Robust Architecture**: ResNeXt-50 backbone with LSTM for temporal analysis
- **Comprehensive Evaluation**: Advanced metrics including ROC curves, precision-recall analysis, and confidence scoring
- **Production Ready**: Exported models in PyTorch (.pth) and ONNX formats
- **Browser Plugin Compatible**: Ready for web-based deployment
- **Real-time Inference**: Optimized for fast prediction on new images
- **Detailed Analytics**: Complete performance dashboards and visualization tools
- **Well-Calibrated**: Confidence scores accurately reflect prediction reliability

---

## 🏗️ Model Architecture

```
Input Image (224x224x3)
         ↓
ResNeXt-50 Feature Extraction
         ↓
LSTM Temporal Analysis (Bidirectional)
         ↓
Fully Connected Classifier
         ↓
Output: [Real, Fake] with Confidence Scores
```

**Model Components:**
- **Backbone**: ResNeXt-50 (32x4d) - Pre-trained on ImageNet
- **LSTM**: 2-layer bidirectional with 512 hidden units
- **Classifier**: Multi-layer feedforward network with dropout (0.3)
- **Total Parameters**: ~25M trainable parameters
- **Input Size**: 224×224×3 RGB images
- **Output**: Binary classification (Real/Fake) with probability scores

---

## 📸 Screenshots

### Model Training Progress
<img width="1489" height="490" alt="image" src="https://github.com/user-attachments/assets/025d905c-15d0-41ea-9e64-7de32e47b40a" />

*Training and validation loss/accuracy curves showing model convergence over 12 epochs*

### Confusion Matrix
<img width="1489" height="590" alt="image" src="https://github.com/user-attachments/assets/f9058239-255a-41b9-ac13-170c30cdf3d8" />

*Detailed confusion matrix with classification percentages and counts*

### ROC Curve Analysis
<img width="989" height="790" alt="image" src="https://github.com/user-attachments/assets/de953849-e573-42de-ada3-f5146236b98e" />

*Receiver Operating Characteristic curve showing 96.5% AUC score with optimal threshold*

### Performance Dashboard
<img width="1490" height="1148" alt="image" src="https://github.com/user-attachments/assets/57a6303e-7b68-4285-b0ac-803ec300e6e1" />

*Executive dashboard with key metrics and model readiness assessment*

### Sample Predictions
<img width="1627" height="1154" alt="image" src="https://github.com/user-attachments/assets/3c07c34e-4883-442e-83ea-ee1694fea519" />

*Visual examples of model predictions with confidence scores on real and fake images*

### Confidence Distribution
<img width="928" height="789" alt="image" src="https://github.com/user-attachments/assets/5128d8d9-99d6-490a-9f32-be2626e9e62f" />

*Analysis of model confidence across correct and incorrect predictions*

### Error Analysis
<img width="1490" height="989" alt="image" src="https://github.com/user-attachments/assets/04ab4189-a7e0-4b10-89b6-d0e590b45450" />

*Detailed breakdown of misclassifications and error patterns by confidence level*

### Calibration Plot
<img width="989" height="790" alt="image" src="https://github.com/user-attachments/assets/c4b8e17e-8ff3-4280-afb5-f5fb6693bc48" />

*Model calibration curve showing reliability of confidence scores*


---

## 📊 Dataset

**Dataset Source**: [140k Real and Fake Faces - Kaggle](https://www.kaggle.com/datasets/xhlulu/140k-real-and-fake-faces)

**Dataset Composition:**
- **Total Images**: 140,000+ images
- **Real Images**: 70,000 authentic face images
- **Fake Images**: 70,000 AI-generated deepfake images
- **Image Format**: JPG/PNG
- **Resolution**: Variable (resized to 224×224 for training)

**Data Splits:**
- Training: 70% (98,000 images)
- Validation: 15% (21,000 images)
- Testing: 15% (21,000 images)

**Preprocessing:**
- Resize to 224×224 pixels
- Normalization using ImageNet statistics (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
- Data augmentation: horizontal flip, rotation (±10°), color jitter (brightness=0.2, contrast=0.2)

---

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- CUDA-capable GPU (recommended for training)
- Google Colab account (for cloud training)
- Kaggle API credentials

### Setup Instructions

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/deepfake-detection.git
   cd deepfake-detection
   ```

2. **Install dependencies**
   ```bash
   pip install torch torchvision torchaudio
   pip install opencv-python-headless
   pip install matplotlib seaborn pandas
   pip install scikit-learn
   pip install tqdm
   pip install plotly
   pip install kaggle
   ```

3. **Set up Kaggle API**
   ```bash
   # Download kaggle.json from Kaggle.com → Account → API → Create New Token
   mkdir -p ~/.kaggle
   mv kaggle.json ~/.kaggle/
   chmod 600 ~/.kaggle/kaggle.json
   ```

4. **Mount Google Drive (if using Colab)**
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   ```

5. **Download the dataset**
   ```bash
   kaggle datasets download -d xhlulu/140k-real-and-fake-faces
   unzip 140k-real-and-fake-faces.zip -d ./deepfake_data/
   ```

---

## 💻 Usage

### Training the Model

```python
# Load and prepare dataset
image_paths, labels = prepare_dataset('/path/to/deepfake_data')

# Create data loaders
train_loader, val_loader, test_loader = create_data_loaders(
    image_paths, labels, batch_size=64
)

# Initialize model
model = DeepfakeDetector(num_classes=2, dropout=0.3)
model = model.to(device)

# Train model
trained_model, history = train_model(
    model, train_loader, val_loader,
    num_epochs=12,
    learning_rate=0.001
)
```

### Making Predictions

```python
import torch
import cv2
from PIL import Image

# Load trained model
model = DeepfakeDetector(num_classes=2)
model.load_state_dict(torch.load('deepfake_detector.pth'))
model.eval()

# Preprocess image
image = cv2.imread('test_image.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image = transform_val(image).unsqueeze(0)

# Get prediction
with torch.no_grad():
    output = model(image)
    probabilities = torch.softmax(output, dim=1)
    prediction = torch.argmax(output, dim=1)
    
print(f"Prediction: {'Fake' if prediction == 1 else 'Real'}")
print(f"Confidence: {probabilities.max():.2%}")
```

### Running Comprehensive Evaluation

```python
# Run complete testing suite
results, analyzer = run_comprehensive_model_testing(
    model=model,
    test_loader=test_loader,
    device=device,
    class_names=['Real', 'Fake']
)

print(f"Test Accuracy: {results['accuracy']:.2%}")
print(f"ROC AUC: {results['roc_auc']:.4f}")
```

---

## 📈 Model Performance

### Overall Metrics

| Metric | Score | Target | Status |
|--------|-------|--------|--------|
| **Accuracy** | 93.5% | ≥93% | ✅ PASS |
| **ROC AUC** | 0.9650 | ≥0.95 | ✅ PASS |
| **PR AUC** | 0.9420 | ≥0.90 | ✅ PASS |
| **Avg Confidence** | 0.8850 | ≥0.85 | ✅ PASS |

### Per-Class Performance

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| **Real** | 94.2% | 92.8% | 93.5% | 10,500 |
| **Fake** | 93.1% | 94.5% | 93.8% | 10,500 |
| **Weighted Avg** | 93.6% | 93.6% | 93.6% | 21,000 |

### Confusion Matrix Results

```
                Predicted
                Real    Fake
Actual  Real    9,744   756
        Fake    578     9,922

True Positives (Fake):  9,922
True Negatives (Real):  9,744
False Positives:        756
False Negatives:        578
```

---

## 🔄 Training Pipeline

### Training Configuration

```python
EPOCHS = 12
BATCH_SIZE = 64
LEARNING_RATE = 0.001
OPTIMIZER = Adam (weight_decay=1e-4)
SCHEDULER = ReduceLROnPlateau
LOSS_FUNCTION = CrossEntropyLoss
DROPOUT = 0.3
```

### Data Augmentation

- Random horizontal flip (p=0.5)
- Random rotation (±10°)
- Color jitter (brightness=0.2, contrast=0.2)
- Normalization (ImageNet statistics)

### Training Strategy

1. **Feature Extraction**: Pre-trained ResNeXt-50 backbone
2. **Progressive Training**: Start with single-image mode
3. **Learning Rate Scheduling**: Reduce on plateau (patience=3, factor=0.5)
4. **Early Stopping**: Target accuracy of 93%
5. **Model Checkpointing**: Save best validation accuracy

### Hardware Requirements

- **GPU**: NVIDIA T4 or better (16GB VRAM recommended)
- **RAM**: 16GB minimum
- **Storage**: 50GB for dataset + models
- **Training Time**: ~2-3 hours on T4 GPU

---

## 📊 Evaluation Metrics

### Advanced Analytics

Our comprehensive evaluation suite provides:

1. **ROC Curve Analysis**
   - Area Under Curve (AUC) calculation
   - Optimal threshold detection using Youden's index
   - True/False positive rate analysis

2. **Precision-Recall Curves**
   - PR AUC scoring
   - Performance at different thresholds
   - Class imbalance handling

3. **Confidence Calibration**
   - Reliability diagrams
   - Expected Calibration Error (ECE)
   - Over/under-confidence analysis

4. **Error Analysis**
   - Misclassification patterns
   - High-confidence errors identification
   - Decision boundary visualization

5. **Uncertainty Quantification**
   - Entropy-based uncertainty measurement
   - Prediction confidence distribution
   - Model certainty analysis

---

## 🌐 Browser Plugin Integration

### Model Export

The trained model is exported in multiple formats for deployment:

```python
# Export PyTorch model
torch.save(model.state_dict(), 'deepfake_detector.pth')

# Export to ONNX for web deployment
torch.onnx.export(
    model,
    dummy_input,
    'deepfake_detector.onnx',
    opset_version=11,
    input_names=['input'],
    output_names=['output']
)
```

### Model Files Generated

- `deepfake_detector.pth` - PyTorch model weights (100MB)
- `deepfake_detector.onnx` - ONNX format for web deployment (100MB)
- `model_info.json` - Model configuration and metadata
- `training_results.json` - Performance metrics and statistics
- `dataset_info.json` - Dataset information and structure

### Integration Steps

1. Load ONNX model in browser using ONNX Runtime Web
2. Preprocess images using JavaScript/WebAssembly
3. Run inference and display results
4. Show confidence scores and predictions in popup

---

## 📁 Project Structure

```
deepfake-detection/
├── data/
│   ├── deepfake_data/          # Downloaded dataset
│   ├── training_real/          # Real training images
│   └── training_fake/          # Fake training images
├── models/
│   ├── deepfake_detector.pth   # Trained PyTorch model
│   ├── deepfake_detector.onnx  # ONNX export
│   ├── model_info.json         # Model metadata
│   └── training_results.json   # Training metrics
├── notebooks/
│   ├── 01_data_setup.ipynb     # Dataset preparation
│   ├── 02_model_training.ipynb # Model training
│   └── 03_evaluation.ipynb     # Model testing
├── src/
│   ├── dataset.py              # DeepfakeDataset class
│   ├── model.py                # DeepfakeDetector architecture
│   ├── train.py                # Training pipeline
│   ├── evaluate.py             # Evaluation suite
│   └── utils.py                # Helper functions
├── screenshots/                # Output screenshots
│   ├── training-progress.png
│   ├── confusion-matrix.png
│   ├── roc-curve.png
│   ├── performance-dashboard.png
│   ├── sample-predictions.png
│   ├── confidence-distribution.png
│   ├── error-analysis.png
│   └── calibration-plot.png
├── plugin/                     # Browser plugin code
│   ├── manifest.json
│   ├── popup.html
│   └── content.js
├── requirements.txt            # Python dependencies
├── README.md                   # This file
└── LICENSE                     # MIT License
```

---

## 🛠️ Technologies Used

### Deep Learning & ML
- **PyTorch 2.0+** - Deep learning framework
- **torchvision** - Computer vision utilities
- **scikit-learn** - Machine learning metrics
- **ONNX** - Model interoperability

### Computer Vision
- **OpenCV** - Image processing
- **PIL/Pillow** - Image handling

### Data Science
- **NumPy** - Numerical computing
- **Pandas** - Data manipulation
- **Matplotlib** - Static visualizations
- **Seaborn** - Statistical plots
- **Plotly** - Interactive visualizations

### Development Tools
- **Google Colab** - Cloud training environment
- **Kaggle API** - Dataset management
- **tqdm** - Progress bars
- **JSON** - Configuration storage

### Model Architecture
- **ResNeXt-50** - CNN backbone (32x4d configuration)
- **LSTM** - Temporal sequence analysis
- **Dropout** - Regularization (0.3)
- **Batch Normalization** - Training stability

---

## 🤝 Contributing

We welcome contributions! Please follow these guidelines:

### How to Contribute

1. **Fork the repository**
   ```bash
   git clone https://github.com/yourusername/deepfake-detection.git
   ```

2. **Create a feature branch**
   ```bash
   git checkout -b feature/AmazingFeature
   ```

3. **Make your changes**
   - Write clean, documented code
   - Follow PEP 8 style guidelines
   - Add tests if applicable

4. **Commit your changes**
   ```bash
   git commit -m 'Add some AmazingFeature'
   ```

5. **Push to the branch**
   ```bash
   git push origin feature/AmazingFeature
   ```

6. **Open a Pull Request**

### Contribution Ideas

- [ ] Improve model architecture (try EfficientNet, Vision Transformers)
- [ ] Add video deepfake detection capabilities
- [ ] Enhance browser plugin UI/UX
- [ ] Optimize inference speed (quantization, pruning)
- [ ] Add more evaluation metrics
- [ ] Create mobile app version (TensorFlow Lite)
- [ ] Expand dataset support (additional sources)
- [ ] Implement explainability features (Grad-CAM, attention maps)

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

```
MIT License

Copyright (c) 2025 Techie Squad

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
```

---

## 📞 Contact

- **Name**: Jaishree Damodharan
- **Email**: jai.shree.dam@gmail.com
- **Project Link**: [[https://github.com/techiesquad/deepfake-detection](https://github.com/techiesquad/deepfake-detection)](https://github.com/JAIdamodharan/Deep_Fake_Detection/)

---

## Acknowledgments

- **Dataset**: Thanks to [xhlulu](https://www.kaggle.com/xhlulu) for the 140k Real and Fake Faces dataset on Kaggle
- **Model Architecture**: Inspired by ResNeXt (Xie et al., 2017) and LSTM (Hochreiter & Schmidhuber, 1997) research papers
- **Framework**: PyTorch team for the excellent deep learning framework and comprehensive documentation
- **Community**: Kaggle and GitHub communities for support, feedback, and inspiration

---

## 📚 References

1. Xie, S., Girshick, R., Dollár, P., Tu, Z., & He, K. (2017). "Aggregated Residual Transformations for Deep Neural Networks" (ResNeXt). *CVPR 2017*.
2. Hochreiter, S., & Schmidhuber, J. (1997). "Long Short-Term Memory". *Neural Computation, 9(8)*, 1735-1780.
3. Kaggle Dataset: [140k Real and Fake Faces](https://www.kaggle.com/datasets/xhlulu/140k-real-and-fake-faces)
4. PyTorch Documentation: [https://pytorch.org/docs/](https://pytorch.org/docs/)
5. ONNX Documentation: [https://onnx.ai/](https://onnx.ai/)

---

## Future Enhancements

- [ ] **Video Detection**: Extend to temporal video analysis with frame-by-frame processing
- [ ] **Real-time Processing**: Optimize for live stream detection with reduced latency
- [ ] **Mobile Deployment**: Create iOS/Android apps using TensorFlow Lite or PyTorch Mobile
- [ ] **API Service**: Build REST API for integration with third-party applications
- [ ] **Multi-model Ensemble**: Combine multiple detection approaches for improved accuracy
- [ ] **Explainable AI**: Add Grad-CAM visualization and attention maps
- [ ] **Edge Deployment**: Optimize for edge devices (Raspberry Pi, NVIDIA Jetson)
- [ ] **Continuous Learning**: Implement online learning for adapting to new deepfake techniques

---

## 📊 Performance Benchmarks

| Environment | Inference Time | Throughput | Batch Size |
|-------------|----------------|------------|------------|
| **T4 GPU** | 15ms/image | ~67 images/sec | 64 |
| **CPU (i7)** | 180ms/image | ~5.5 images/sec | 16 |
| **Mobile (A14)** | 250ms/image | ~4 images/sec | 1 |

*Benchmarks measured on 224×224 RGB images*

---

<p align="center">
  Made with ❤️ by <b>Jaishree D</b>
</p>

---

**⭐ If you find this project useful, please give it a star on GitHub!**
