# ✍️ Handwritten Digit Recognition (Deep Learning CNN)

Deep Learning Convolutional Neural Network (CNN) var aadharit Handwritten Digit Recognition project. Ha project Google Colab code la local Windows machine var run honyasathi optimize kela ahe.

---

## 📁 Project Structure

```
handwritten_digit/
│
├── dataset/                  # Custom dataset folder (images 0_*.jpg etc.)
│   └── README.md
├── templates/
│   └── index.html            # Web app UI template
├── static/
│   ├── style.css             # Dark mode glassmorphism UI styles
│   └── script.js             # Interactive drawing canvas & API logic
│
├── train.py                  # Model training script
├── predict.py                # Command-line image prediction script
├── app.py                    # Flask Web Application (Canvas + Upload)
├── requirements.txt          # Python dependencies
└── README.md                 # Project guide
```

---

## ⚡ Quick Start (कसे चालवायचे)

### 1. Requirements Install करा
Terminal / Command Prompt open kara ani khali dili command run kara:

```bash
pip install -r requirements.txt
```

---

### 2. Model Train करा (`train.py`)

```bash
python train.py
```
**Features of `train.py`:**
- Jar `HANDWRITTEN DIGITS DATASET.zip` file asel tar ti auto-extract hote.
- Jar custom dataset asel (`0_*.jpg`, `1_*.png`, etc.), tar te images preprocess karel.
- **Auto Fallback**: Jar custom dataset sapadla nahi, tar automatic standard **MNIST Dataset** load hoto ani training complete hote.
- Training purna zalyavar `model.keras`, `model.h5`, ani `accuracy_loss_plot.png` save hoto.

---

### 3. Image Predict करा (`predict.py`)

Konthya hi ekhadya digit chya image var test karnyasathi:

```bash
python predict.py path/to/digit_image.png
```
Or just run `python predict.py` and it will ask you to enter the image path.

---

### 4. Interactive Web App चालवा (`app.py`) 🚀

Mouse/Finger ne digit draw karun live prediction pahnyasathi web app start kara:

```bash
python app.py
```

Nantar browser madhe open kara:
👉 **[http://127.0.0.1:5000](http://127.0.0.1:5000)**

**Web App Features:**
- ✏️ **Live Drawing Canvas**: Mouse ne 0 te 9 digit draw kara ani **Predict** dabun result paha.
- 📁 **Image Upload**: Computer madhun digit chi photo upload karun predict kara.
- 📊 **Probabilities Breakdown**: 0 te 9 sarv digits che confidence % chart disel.

---

## 🧠 Model Architecture (CNN)

1. **Input**: `(28, 28, 1)` Grayscale Image (Inverted & Normalized)
2. **Conv2D**: 32 Filters (3x3), ReLU
3. **MaxPooling2D**: (2x2)
4. **Conv2D**: 64 Filters (3x3), ReLU
5. **MaxPooling2D**: (2x2)
6. **Flatten**
7. **Dense**: 128 Neurons, ReLU
8. **Dense (Output)**: 10 Neurons (Digits 0-9), Softmax

---

## 👨‍💻 Author

**Rohan Shinde**
- Handwritten Digit Recognition AI & Deep Learning Web App
