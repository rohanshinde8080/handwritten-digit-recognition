import os
import zipfile
import argparse
import numpy as np
try:
    import cv2  # type: ignore
except (ImportError, ModuleNotFoundError):
    cv2 = None
from PIL import Image
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models

def extract_zip_if_exists(zip_filename="HANDWRITTEN DIGITS DATASET.zip", extract_to="dataset"):
    """Extract zip dataset if available."""
    if os.path.exists(zip_filename):
        print(f"📦 Found '{zip_filename}'. Extracting to '{extract_to}'...")
        os.makedirs(extract_to, exist_ok=True)
        with zipfile.ZipFile(zip_filename, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
        print("✅ Dataset Extracted Successfully.")

def load_custom_dataset(base_path):
    """Load and preprocess images from custom dataset folder."""
    data = []
    labels = []

    if not os.path.exists(base_path):
        return None, None

    # Search in base_path and subdirectories
    for root, _, files in os.walk(base_path):
        for file_name in files:
            if file_name.lower().endswith(('.jpeg', '.png', '.jpg')):
                try:
                    # Expecting file format like '0_123.png' or '7_img.jpg'
                    label_str = file_name.split('_')[0]
                    label = int(label_str)
                    if label < 0 or label > 9:
                        continue
                except ValueError:
                    # If not split by underscore, check if folder name is digit
                    folder_name = os.path.basename(root)
                    if folder_name.isdigit():
                        label = int(folder_name)
                    else:
                        print(f"⚠️ Skipping file with unparsable label: {file_name}")
                        continue

                path = os.path.join(root, file_name)
                if cv2 is not None:
                    img = cv2.imread(path)
                    if img is None:
                        print(f"⚠️ Warning: Could not read image {path}. Skipping.")
                        continue
                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    gray = 255 - gray
                    _, gray = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
                    resized = cv2.resize(gray, (28, 28))
                    normalized = resized / 255.0
                else:
                    try:
                        pil_img = Image.open(path).convert('L')
                        gray = 255 - np.array(pil_img)
                        gray = (gray > 150).astype(np.uint8) * 255
                        pil_resized = Image.fromarray(gray).resize((28, 28), Image.Resampling.BILINEAR)
                        normalized = np.array(pil_resized, dtype=np.float32) / 255.0
                    except Exception:
                        print(f"⚠️ Warning: Could not read image {path}. Skipping.")
                        continue

                data.append(normalized)
                labels.append(label)

    if len(data) == 0:
        return None, None

    data = np.array(data, dtype=np.float32).reshape(-1, 28, 28, 1)
    labels = np.array(labels, dtype=np.int64)
    return data, labels

def build_cnn_model():
    """Build Convolutional Neural Network model for digit classification."""
    model = models.Sequential([
        layers.Input(shape=(28, 28, 1)),
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),

        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

def main():
    parser = argparse.ArgumentParser(description="Train Handwritten Digit Recognition Model")
    parser.add_argument("--dataset_dir", type=str, default="dataset", help="Path to dataset directory")
    parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Training batch size")
    args = parser.parse_args()

    # Step 1: Check zip extraction
    extract_zip_if_exists("HANDWRITTEN DIGITS DATASET.zip", args.dataset_dir)
    extract_zip_if_exists("dataset.zip", args.dataset_dir)

    # Step 2: Search for dataset in possible directories
    possible_paths = [
        args.dataset_dir,
        "HANDWRITTEN DIGITS DATASET",
        os.path.join(args.dataset_dir, "HANDWRITTEN DIGITS DATASET")
    ]

    data, labels = None, None
    for p in possible_paths:
        if os.path.exists(p):
            print(f"🔍 Searching for images in '{p}'...")
            data, labels = load_custom_dataset(p)
            if data is not None and len(data) > 0:
                print(f"✅ Loaded {len(data)} images from '{p}'")
                break

    # Fallback to standard MNIST dataset if no custom dataset found
    if data is None or len(data) == 0:
        print("\nℹ️ No custom dataset images found in local folders.")
        print("📥 Automatically loading standard MNIST dataset as fallback...")
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
        data = (x_train / 255.0).reshape(-1, 28, 28, 1)
        labels = y_train
        print(f"✅ Loaded MNIST dataset with shape: {data.shape}")

    print(f"\n📊 Total Training Samples: {data.shape[0]}")
    print(f"📐 Input Shape: {data.shape[1:]}")

    # Step 3: Build & Train Model
    print("\n🧠 Building CNN Model...")
    model = build_cnn_model()
    model.summary()

    print(f"\n🚀 Training model for {args.epochs} epochs...")
    history = model.fit(
        data, 
        labels, 
        epochs=args.epochs, 
        batch_size=args.batch_size,
        validation_split=0.1 if len(data) >= 100 else 0.0
    )
    print("✅ Training Done!")

    # Step 4: Save Model
    model.save("model.keras")
    try:
        model.save("model.h5")
    except Exception:
        pass
    print("💾 Model saved as 'model.keras' and 'model.h5'")

    # Step 5: Save Plot
    plt.figure(figsize=(10, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy', color='blue')
    if 'val_accuracy' in history.history:
        plt.plot(history.history['val_accuracy'], label='Val Accuracy', color='green')
    plt.title("Model Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss', color='red')
    if 'val_loss' in history.history:
        plt.plot(history.history['val_loss'], label='Val Loss', color='orange')
    plt.title("Model Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)

    plot_file = "accuracy_loss_plot.png"
    plt.tight_layout()
    plt.savefig(plot_file, dpi=150)
    plt.close()
    print(f"📈 Training curve saved to '{plot_file}'")

if __name__ == "__main__":
    main()
