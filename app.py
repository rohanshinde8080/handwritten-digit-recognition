import os
import io
import re
import base64
import numpy as np
try:
    import cv2  # type: ignore
except (ImportError, ModuleNotFoundError):
    cv2 = None
from PIL import Image
try:
    import tensorflow as tf  # type: ignore
except (ImportError, ModuleNotFoundError):
    tf = None

app = Flask(__name__)

# Global model variable
model = None

def get_model():
    global model
    if model is None and tf is not None:
        for path in ["model.keras", "model.h5"]:
            if os.path.exists(path):
                print(f"Loading model from {path}...")
                model = tf.keras.models.load_model(path)
                break
    return model

def preprocess_numpy_image(img_array):
    """
    Robust preprocessing for both canvas drawings and uploaded real-world photos.
    - Handles uneven shadows, lighting gradients, and paper backgrounds (Adaptive Thresholding)
    - Filters out photo frame borders, desk/table edges, and camera noise
    - Ensures stroke boldness and aspect-ratio preservation (20x20 box inside 28x28)
    - Aligns digit by Center of Mass (Standard MNIST convention)
    """
    if cv2 is not None:
        # Resize huge photos down to max dim 500 for consistent thresholding and fast execution
        h_orig, w_orig = img_array.shape[:2]
        max_dim = max(h_orig, w_orig)
        if max_dim > 500:
            scale = 500.0 / max_dim
            img_array = cv2.resize(img_array, (int(w_orig * scale), int(h_orig * scale)), interpolation=cv2.INTER_AREA)

        # Handle color channels
        if len(img_array.shape) == 3:
            if img_array.shape[2] == 4:  # RGBA
                alpha = img_array[:, :, 3]
                rgb = img_array[:, :, :3]
                if np.min(alpha) < 250:
                    gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
                    if np.max(gray) < 50 and np.max(alpha) > 50:
                        gray = alpha
                    else:
                        alpha_norm = (alpha.astype(np.float32) / 255.0)[:, :, np.newaxis]
                        blended = (rgb.astype(np.float32) * alpha_norm).astype(np.uint8)
                        gray = cv2.cvtColor(blended, cv2.COLOR_BGR2GRAY)
                else:
                    gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
            else:
                gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
        else:
            gray = img_array.copy()

        h, w = gray.shape

        # Detect background color using image borders
        border_pixels = np.concatenate([
            gray[:max(2, int(h*0.05)), :].ravel(),
            gray[-max(2, int(h*0.05)):, :].ravel(),
            gray[:, :max(2, int(w*0.05))].ravel(),
            gray[:, -max(2, int(w*0.05)):].ravel()
        ])
        is_light_bg = np.median(border_pixels) > 115 or np.mean(gray) > 120

        if is_light_bg:
            # Real photo / paper: Gaussian blur + Adaptive Gaussian Thresholding to eliminate shadows
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            thresh = cv2.adaptiveThreshold(
                blurred, 255, 
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                cv2.THRESH_BINARY_INV, 
                25, 12
            )

            # Suppress 4% image border to eliminate photo frame / paper edge shadows
            b_h = max(2, int(h * 0.04))
            b_w = max(2, int(w * 0.04))
            thresh[:b_h, :] = 0
            thresh[-b_h:, :] = 0
            thresh[:, :b_w] = 0
            thresh[:, -b_w:] = 0

            # Morphological noise cleanup
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

            # Background subtraction to extract genuine ink stroke anti-aliasing
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
            # Dark canvas background
            _, thresh = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY)
            digit_src = thresh

        # Find digit contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        total_area = h * w
        valid_boxes = []
        for c in contours:
            area = cv2.contourArea(c)
            # Discard tiny speckles (< 0.05% of image area)
            if area > total_area * 0.0005:
                valid_boxes.append(cv2.boundingRect(c))

        if valid_boxes:
            # Add small padding around bounding box
            pad = 2
            x1 = max(0, min(b[0] for b in valid_boxes) - pad)
            y1 = max(0, min(b[1] for b in valid_boxes) - pad)
            x2 = min(w, max(b[0] + b[2] for b in valid_boxes) + pad)
            y2 = min(h, max(b[1] + b[3] for b in valid_boxes) + pad)
            digit_crop = digit_src[y1:y2, x1:x2]
        else:
            digit_crop = digit_src

        # Maintain aspect ratio inside 20x20 box
        ch, cw = digit_crop.shape[:2]
        if ch > 0 and cw > 0:
            if cw > ch:
                new_w = 20
                new_h = max(1, int(round(ch * (20.0 / cw))))
            else:
                new_h = 20
                new_w = max(1, int(round(cw * (20.0 / ch))))
            
            resized_digit = cv2.resize(digit_crop, (new_w, new_h), interpolation=cv2.INTER_AREA)

            # Check if strokes are too thin (dilate slightly to match MNIST stroke boldness)
            if np.mean(resized_digit > 40) < 0.16:
                resized_digit = cv2.dilate(resized_digit, np.ones((2, 2), np.uint8), iterations=1)

            canvas28 = np.zeros((28, 28), dtype=np.uint8)
            pad_top = (28 - new_h) // 2
            pad_left = (28 - new_w) // 2
            canvas28[pad_top:pad_top+new_h, pad_left:pad_left+new_w] = resized_digit

            # Center of Mass adjustment (standard MNIST alignment)
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
        # Pure PIL/Numpy fallback
        if len(img_array.shape) == 3:
            pil_img = Image.fromarray(img_array).convert('L')
        else:
            pil_img = Image.fromarray(img_array)

        # Scale down if very large
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
    tensor = normalized.reshape(1, 28, 28, 1)
    return tensor

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict-canvas', methods=['POST'])
def predict_canvas():
    try:
        active_model = get_model()
        if active_model is None:
            return jsonify({'error': 'Model not found. Please run python train.py first!'}), 400

        data = request.get_json()
        if not data or 'image' not in data:
            return jsonify({'error': 'No image data received'}), 400

        # Decode base64 image
        image_data = re.sub('^data:image/.+;base64,', '', data['image'])
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes))
        img_array = np.array(image)

        tensor = preprocess_numpy_image(img_array)
        preds = active_model.predict(tensor, verbose=0)[0]

        pred_digit = int(np.argmax(preds))
        confidence = float(preds[pred_digit]) * 100
        probabilities = [float(p) * 100 for p in preds]

        return jsonify({
            'success': True,
            'digit': pred_digit,
            'confidence': round(confidence, 2),
            'probabilities': [round(p, 2) for p in probabilities]
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/predict-upload', methods=['POST'])
def predict_upload():
    try:
        active_model = get_model()
        if active_model is None:
            return jsonify({'error': 'Model not found. Please run python train.py first!'}), 400

        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400

        image = Image.open(file.stream)
        img_array = np.array(image)

        tensor = preprocess_numpy_image(img_array)
        preds = active_model.predict(tensor, verbose=0)[0]

        pred_digit = int(np.argmax(preds))
        confidence = float(preds[pred_digit]) * 100
        probabilities = [float(p) * 100 for p in preds]

        return jsonify({
            'success': True,
            'digit': pred_digit,
            'confidence': round(confidence, 2),
            'probabilities': [round(p, 2) for p in probabilities]
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("🚀 Starting Handwritten Digit Recognition Web App on http://127.0.0.1:5000")
    app.run(host='0.0.0.0', port=5000, debug=True)
