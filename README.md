# ✍️ Handwritten Digit Recognition (Deep Learning CNN)

An end-to-end Deep Learning Web Application for real-time **Handwritten Digit Recognition (0–9)** powered by a Convolutional Neural Network (CNN) built with **TensorFlow / Keras**, **OpenCV**, and **Flask**.

---

## 🌟 Features

- ✏️ **Interactive Drawing Canvas**: Draw any digit (0–9) directly in your browser with adjustable brush sizes.
- 📸 **Real-World Photo Upload**: Upload photographs of handwritten digits on paper. Built-in **Adaptive Gaussian Thresholding** automatically eliminates paper grain, uneven room lighting, and camera shadows.
- 🎯 **High Accuracy CNN (~99.3% Test Accuracy)**: Custom deep convolutional neural network trained on the MNIST dataset with data normalization and centering.
- 📊 **Real-Time Probability Breakdown**: Dynamic probability bar charts showing confidence scores across all 10 classes (0 to 9).
- 🎨 **Modern Glassmorphism UI**: Beautiful dark-mode interface with smooth animations, glow effects, and mobile-friendly touch support.
- ⚡ **Center-of-Mass Preprocessing**: Automatically aligns handwritten inputs using image moments ($14, 14$) to match MNIST training standards.

---

## 📁 Project Structure

```
handwritten_digit/
│
├── dataset/                  # Custom dataset directory (optional)
│   └── README.md
├── templates/
│   └── index.html            # Frontend HTML5 template
├── static/
│   ├── style.css             # Glassmorphism dark mode CSS styling
│   └── script.js             # Canvas drawing logic & API integrations
│
├── train.py                  # Model training pipeline (Custom + MNIST fallback)
├── predict.py                # Command-line image prediction tool
├── app.py                    # Flask Web Server (REST API endpoints)
├── model.keras               # Trained Keras CNN model weights
├── model.h5                  # Legacy HDF5 model format
├── accuracy_loss_plot.png    # Training accuracy & loss curves
├── Procfile                  # Production configuration for Cloud hosting
├── requirements.txt          # Python project dependencies
├── .python-version           # Cloud Python version definition (3.10.14)
└── README.md                 # Project documentation
```

---

## 🛠️ Tech Stack

- **Backend & AI**: Python 3.10, TensorFlow / Keras, OpenCV (`cv2`), Pillow (PIL), NumPy
- **Web Framework**: Flask, Gunicorn
- **Frontend**: HTML5, CSS3 (Modern Glassmorphism), JavaScript (ES6+ Canvas API)
- **Deployment**: Render.com, Git & GitHub

---

## ⚡ Quick Start Guide

### 1. Clone the Repository & Install Dependencies

```bash
git clone https://github.com/rohanshinde8080/handwritten-digit-recognition.git
cd handwritten-digit-recognition
pip install -r requirements.txt
```

---

### 2. Train the Model (`train.py`)

Train a new CNN model using custom images or standard MNIST:

```bash
python train.py
```

**Key Training Features:**
- Automatically extracts `HANDWRITTEN DIGITS DATASET.zip` if present.
- Supports custom image directories with `label_*.jpg` format or subdirectories `0/`, `1/`, etc.
- Automatically falls back to the official **MNIST Dataset** (60,000 samples) if no local dataset is found.
- Saves the best weights to `model.keras`, `model.h5`, and outputs `accuracy_loss_plot.png`.

---

### 3. Predict from Command Line (`predict.py`)

Test prediction on any image directly in the terminal:

```bash
python predict.py path/to/digit_image.png
```

Or run interactively:
```bash
python predict.py
```

---

### 4. Run the Web Application Locally (`app.py`) 🚀

Start the Flask development server:

```bash
python app.py
```

Then open your browser and navigate to:
👉 **`http://127.0.0.1:5000`**

---

## 🧠 Convolutional Neural Network (CNN) Architecture

| Layer | Type | Specifications | Output Shape |
| :--- | :--- | :--- | :--- |
| **Input** | InputLayer | Grayscale Normalized Matrix | `(28, 28, 1)` |
| **Layer 1** | Conv2D | 32 Filters, 3×3 Kernel, ReLU | `(26, 26, 32)` |
| **Layer 2** | MaxPooling2D | 2×2 Pool Size | `(13, 13, 32)` |
| **Layer 3** | Conv2D | 64 Filters, 3×3 Kernel, ReLU | `(11, 11, 64)` |
| **Layer 4** | MaxPooling2D | 2×2 Pool Size | `(5, 5, 64)` |
| **Layer 5** | Flatten | Vector Reshape | `(1600)` |
| **Layer 6** | Dense | 128 Neurons, ReLU | `(128)` |
| **Output** | Dense | 10 Neurons (Classes 0–9), Softmax | `(10)` |

- **Loss Function**: `sparse_categorical_crossentropy`
- **Optimizer**: `Adam`
- **Validation Accuracy**: **~99.33%**

---

## 🌐 Live Cloud Deployment

This project is configured for one-click deployment on **Render.com**:
- **Build Command**: `pip install -r requirements.txt`
- **Start Command**: `gunicorn app:app`

---

## 👨‍💻 Author

**Rohan Shinde**
- Project: *Handwritten Digit Recognition AI & Deep Learning Web App*
- GitHub: [@rohanshinde8080](https://github.com/rohanshinde8080)
