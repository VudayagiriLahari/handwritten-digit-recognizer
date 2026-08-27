# Handwritten Digit Recognizer (CNN + MNIST)

A deep learning project that trains a Convolutional Neural Network (CNN) to classify handwritten digits (0-9) using the MNIST dataset, achieving over 99% test accuracy.

---

## Features

| Feature | Detail |
|---------|--------|
| Dataset | MNIST (70,000 images, 28x28 grayscale) |
| Model | 2-block CNN with BatchNorm and Dropout |
| Augmentation | Rotation, shift, zoom |
| Accuracy | ~99.2% on test set |
| Outputs | Training curves, confusion matrix, feature maps, misclassifications |

---

## Setup

```bash
git clone https://github.com/VudayagiriLahari/handwritten-digit-recognizer.git
cd handwritten-digit-recognizer

pip install -r requirements.txt

python digit_recognizer.py
```

---

## Requirements

```
numpy
matplotlib
seaborn
tensorflow>=2.13
scikit-learn
```

---

## Model Architecture

```
Input (28x28x1)
    |
Conv2D(32) -> BatchNorm -> ReLU
Conv2D(32) -> BatchNorm -> ReLU
    |
MaxPool(2x2) -> Dropout(0.25)
    |
Conv2D(64) -> BatchNorm -> ReLU
Conv2D(64) -> BatchNorm -> ReLU
    |
MaxPool(2x2) -> Dropout(0.25)
    |
Flatten
    |
Dense(256) -> BatchNorm -> ReLU -> Dropout(0.5)
    |
Dense(10) -> Softmax
```

Total parameters: ~420,000

---

## Training Details

| Setting | Value |
|---------|-------|
| Optimizer | Adam (lr=0.001) |
| Loss | Categorical Cross-Entropy |
| Batch size | 128 |
| Max epochs | 20 |
| EarlyStopping | patience=5 |
| LR Scheduler | ReduceLROnPlateau |
| Augmentation | Rotation 10deg, shift 10%, zoom 10% |

---

## Output Files

```
results/
├── mnist_cnn.keras         # Saved best model
├── training_curves.png     # Accuracy and loss over epochs
├── confusion_matrix.png    # 10x10 heatmap
├── sample_predictions.png  # 36-image grid
├── feature_maps.png        # Conv1 feature map visualisation
└── misclassified.png       # Samples the model got wrong
```

---

## Results

| Metric | Score |
|--------|-------|
| Test Accuracy | ~99.2% |
| Test Loss | ~0.025 |

Per-class precision and recall are printed to console after training.

---

## How to Predict on a Custom Image

```python
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np

model = load_model("results/mnist_cnn.keras")

img = Image.open("my_digit.png").convert("L").resize((28, 28))
arr = np.array(img, dtype="float32") / 255.0
arr = arr.reshape(1, 28, 28, 1)

prediction = np.argmax(model.predict(arr))
print(f"Predicted digit: {prediction}")
```

---

## Tech Stack

Python · TensorFlow/Keras · NumPy · Matplotlib · Seaborn · scikit-learn
