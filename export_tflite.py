import os
import tensorflow as tf

def convert_to_tflite():
    for model_path in ["model.keras", "model.h5"]:
        if os.path.exists(model_path):
            print(f"Loading {model_path} for TFLite conversion...")
            model = tf.keras.models.load_model(model_path)
            converter = tf.lite.TFLiteConverter.from_keras_model(model)
            tflite_model = converter.convert()
            with open("model.tflite", "wb") as f:
                f.write(tflite_model)
            print(f"✅ Successfully exported model.tflite (Size: {len(tflite_model)/1024:.1f} KB)")
            return

if __name__ == "__main__":
    convert_to_tflite()
