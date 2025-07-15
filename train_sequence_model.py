# train_sequence_model.py
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import TimeDistributed, LSTM, Dense, Input
from sklearn.model_selection import train_test_split
import random

SEQUENCE_LENGTH = 16
IMG_SIZE = (250, 250)
CLASS_MAP = {"Accident": 0, "No Accident": 1}

# --- Load sequences ---
def load_sequences_from_directory(base_path):
    sequences = []
    labels = []

    for class_label in os.listdir(base_path):
        class_path = os.path.join(base_path, class_label)
        all_frames = sorted([os.path.join(class_path, f) for f in os.listdir(class_path) if f.endswith(".jpg")])

        # Create sequences of SEQUENCE_LENGTH frames
        for i in range(0, len(all_frames) - SEQUENCE_LENGTH + 1, SEQUENCE_LENGTH):
            sequence = []
            for j in range(i, i + SEQUENCE_LENGTH):
                img = tf.keras.preprocessing.image.load_img(all_frames[j], target_size=IMG_SIZE)
                img_array = tf.keras.preprocessing.image.img_to_array(img) / 255.0
                sequence.append(img_array)
            sequences.append(sequence)
            labels.append(CLASS_MAP[class_label])

    return np.array(sequences), np.array(labels)

# Load data
print("[INFO] Loading sequences...")
x, y = load_sequences_from_directory("dataset/train")
print(f"[INFO] Loaded {len(x)} sequences.")

# Split data
x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=42)

# --- Build model ---
input_layer = Input(shape=(SEQUENCE_LENGTH, IMG_SIZE[0], IMG_SIZE[1], 3))

cnn_base = ResNet50(weights='imagenet', include_top=False, pooling='avg', input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
cnn_base.trainable = False

x = TimeDistributed(cnn_base)(input_layer)
x = LSTM(64)(x)
x = Dense(64, activation='relu')(x)
output = Dense(2, activation='softmax')(x)

model = models.Model(inputs=input_layer, outputs=output)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()

# --- Train model ---
model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=10, batch_size=4)

# --- Save model ---
model_json = model.to_json()
with open("model/sequence_model.json", "w") as json_file:
    json_file.write(model_json)
model.save_weights("model/sequence_model.weights.h5")
print("[✔] Sequence model saved.")