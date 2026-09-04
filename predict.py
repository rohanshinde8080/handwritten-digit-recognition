import os
import sys
import argparse
import numpy as np
try:
    import cv2  # type: ignore
except (ImportError, ModuleNotFoundError):
    cv2 = None
from PIL import Image
import matplotlib.pyplot as plt
import tensorflow as tf

def preprocess_image(image_path):
    """Preprocess single image for model prediction with adaptive thresholding and MNIST-style centering."""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found at path: {image_path}")

    if cv2 is not None:
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not open or find the image at: {image_path}")

        h_orig, w_orig = img.shape[:2]
        max_dim = max(h_orig, w_orig)
        if max_dim > 500:
            scale = 500.0 / max_dim
            img = cv2.resize(img, (int(w_orig * scale), int(h_orig * scale)), interpolation=cv2.INTER_AREA)

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape

        border_pixels = np.concatenate([
            gray[:max(2, int(h*0.05)), :].ravel(),
            gray[-max(2, int(h*0.05)):, :].ravel(),
            gray[:, :max(2, int(w*0.05))].ravel(),
            gray[:, -max(2, int(w*0.05)):].ravel()
        ])
        is_light_bg = np.median(border_pixels) > 115 or np.mean(gray) > 120

        if is_light_bg:
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            thresh = cv2.adaptiveThreshold(
                blurred, 255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY_INV,
                25, 12
            )
            b_h = max(2, int(h * 0.04))
            b_w = max(2, int(w * 0.04))
            thresh[:b_h, :] = 0
            thresh[-b_h:, :] = 0
            thresh[:, :b_w] = 0
            thresh[:, -b_w:] = 0

            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

            # Background subtraction for genuine ink anti-aliasing
            bg_est = cv2.GaussianBlur(gray, (41, 41), 0)
            ink = cv2.subtract(bg_est, gray)
            ink = cv2.bitwise_and(ink, ink, mask=thresh)

            ink_max = np.max(ink)
            if ink_max > 25:
                ink = ((ink.astype(np.float32) / ink_max) * 255.0).astype(np.uint8)
            else:
                ink = thresh
            digit_src = ink
        else:
            _, thresh = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY)
            digit_src = thresh

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        total_area = h * w
        valid_boxes = []
        for c in contours:
            area = cv2.contourArea(c)
            if area > total_area * 0.0005:
                valid_boxes.append(cv2.boundingRect(c))

        if valid_boxes:
            pad = 2
            x1 = max(0, min(b[0] for b in valid_boxes) - pad)
            y1 = max(0, min(b[1] for b in valid_boxes) - pad)
            x2 = min(w, max(b[0] + b[2] for b in valid_boxes) + pad)
            y2 = min(h, max(b[1] + b[3] for b in valid_boxes) + pad)
            digit_crop = digit_src[y1:y2, x1:x2]
        else:
            digit_crop = digit_src

        ch, cw = digit_crop.shape[:2]
        if ch > 0 and cw > 0:
            if cw > ch:
                new_w = 20
                new_h = max(1, int(round(ch * (20.0 / cw))))
            else:
                new_h = 20
                new_w = max(1, int(round(cw * (20.0 / h))))

            resized_digit = cv2.resize(digit_crop, (new_w, new_h), interpolation=cv2.INTER_AREA)

            if np.mean(resized_digit > 40) < 0.16:
                resized_digit = cv2.dilate(resized_digit, np.ones((2, 2), np.uint8), iterations=1)

            canvas28 = np.zeros((28, 28), dtype=np.uint8)
            pad_top = (28 - new_h) // 2
            pad_left = (28 - new_w) // 2
            canvas28[pad_top:pad_top+new_h, pad_left:pad_left+new_w] = resized_digit

            M = cv2.moments(canvas28)
            if M["m00"] > 0:
                cx = M["m10"] / M["m00"]
                cy = M["m01"] / M["m00"]
                shift_x = int(round(14 - cx))
                shift_y = int(round(14 - cy))
                shift_x = max(-3, min(3, shift_x))
                shift_y = max(-3, min(3, shift_y))
                M_shift = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
                canvas28 = cv2.warpAffine(canvas28, M_shift, (28, 28), flags=cv2.INTER_LINEAR, borderValue=0)

            processed = canvas28
        else:
            processed = cv2.resize(gray, (28, 28), interpolation=cv2.INTER_AREA)
    else:
        pil_img = Image.open(image_path).convert('L')
        max_dim = max(pil_img.size)
        if max_dim > 500:
            scale = 500.0 / max_dim
            pil_img = pil_img.resize((int(pil_img.width * scale), int(pil_img.height * scale)), Image.Resampling.BILINEAR)

        gray = np.array(pil_img)
        border_pixels = np.concatenate([gray[0, :], gray[-1, :], gray[:, 0], gray[:, -1]])
        is_light_bg = np.median(border_pixels) > 115 or np.mean(gray) > 120

        if is_light_bg:
            gray = 255 - gray
            thresh = (gray > (np.mean(gray) + 15)).astype(np.uint8) * 255
        else:
            thresh = (gray > 30).astype(np.uint8) * 255

        nonzero = np.argwhere(thresh > 0)
        if len(nonzero) > 10:
            y_min, x_min = nonzero.min(axis=0)
            y_max, x_max = nonzero.max(axis=0)
            w = x_max - x_min + 1
            h = y_max - y_min + 1
            digit_crop = Image.fromarray(thresh[y_min:y_max+1, x_min:x_max+1])
            if w > h:
                new_w = 20
                new_h = max(1, int(round(h * (20.0 / w))))
            else:
                new_h = 20
                new_w = max(1, int(round(w * (20.0 / h))))
            resized = digit_crop.resize((new_w, new_h), Image.Resampling.LANCZOS)
            canvas28 = Image.new('L', (28, 28), 0)
            canvas28.paste(resized, ((28 - new_w) // 2, (28 - new_h) // 2))
            processed = np.array(canvas28)
        else:
            processed = np.array(pil_img.resize((28, 28), Image.Resampling.LANCZOS))

    normalized = processed.astype(np.float32) / 255.0
    reshaped = normalized.reshape(1, 28, 28, 1)

    return reshaped, processed

def load_trained_model():
    """Load model from model.keras or model.h5."""
    for model_path in ["model.keras", "model.h5"]:
        if os.path.exists(model_path):
            print(f"📦 Loading model from '{model_path}'...")
            return tf.keras.models.load_model(model_path)
    raise FileNotFoundError("Model file ('model.keras' or 'model.h5') not found. Please run 'python train.py' first.")

def main():
    parser = argparse.ArgumentParser(description="Predict handwritten digit from an image")
    parser.add_argument("image_path", nargs="?", default=None, help="Path to input digit image")
    parser.add_argument("--save_preview", action="store_true", help="Save preprocessed image as 'prediction_preview.png'")
    args = parser.parse_args()

    image_path = args.image_path
    if not image_path:
        image_path = input("Enter path to digit image (e.g. test.png): ").strip().strip('"')

    if not os.path.exists(image_path):
        print(f"❌ Error: File '{image_path}' does not exist.")
        sys.exit(1)

    try:
        model = load_trained_model()
        tensor_img, preview_img = preprocess_image(image_path)

        predictions = model.predict(tensor_img, verbose=0)[0]
        predicted_digit = int(np.argmax(predictions))
        confidence = float(predictions[predicted_digit]) * 100

        print("\n" + "="*40)
        print(f"🎯 PREDICTED DIGIT : {predicted_digit}")
        print(f"📊 CONFIDENCE       : {confidence:.2f}%")
        print("="*40)

        # Top 3 probabilities
        top_indices = np.argsort(predictions)[::-1][:3]
        print("\nTop 3 Predictions:")
        for idx in top_indices:
            print(f"  Digit {idx}: {predictions[idx]*100:.2f}%")

        if args.save_preview:
            preview_file = "prediction_preview.png"
            cv2.imwrite(preview_file, preview_img)
            print(f"\n🖼️ Preprocessed 28x28 preview saved to '{preview_file}'")

    except Exception as e:
        print(f"❌ Error during prediction: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
