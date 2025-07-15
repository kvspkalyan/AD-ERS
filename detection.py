import numpy as np
import cv2
from tensorflow.keras.models import model_from_json

def load_model():
    with open("model/mobilenet_model.json", "r") as f:
        model = model_from_json(f.read())
    model.load_weights("model/mobilenet_model.weights.h5")
    print("[INFO] Model loaded.")
    return model

def preprocess_frame(frame):
    img = cv2.resize(frame, (128, 128))
    img = img.astype(np.float32) / 255.0
    return np.expand_dims(img, axis=0)

def predict_frame(model, frame, class_names):
    processed = preprocess_frame(frame)
    preds = model.predict(processed, verbose=0)
    print("Prediction:", preds[0])
    index = np.argmax(preds[0])
    return class_names[index], preds[0][index]
