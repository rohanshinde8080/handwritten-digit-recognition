# 📁 Dataset Directory

You can place your custom handwritten digit image dataset in this folder.

### 📝 Supported Formats & Naming Conventions:

#### Format A: Filename with Label Prefix
Each image file should begin with the target digit label followed by an underscore:
- `0_sample1.jpg`
- `0_img2.png`
- `1_digit.png`
- `7_test.jpeg`
- `9_custom.jpg`

#### Format B: Class Subfolders
Organize images into individual subfolders for each digit:
```
dataset/
├── 0/
├── 1/
├── 2/
...
└── 9/
```

### ⚡ Automatic Fallback & Zip Extraction:
- If you place a `HANDWRITTEN DIGITS DATASET.zip` or `dataset.zip` file in the main project folder, running `python train.py` will automatically extract it.
- If this dataset directory is empty, `train.py` will automatically load the official **MNIST Dataset** (60,000 training samples) and train the model.
