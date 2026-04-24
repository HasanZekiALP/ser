import os
import numpy as np
import librosa
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report

# Veri seti klasörü (RAVDESS)
DATASET_PATH = "ravdess/"

def extract_features(file_path):
    y, sr = librosa.load(file_path, sr=None)
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    mel_spec_db = (mel_spec_db - np.mean(mel_spec_db)) / np.std(mel_spec_db)
    return mel_spec_db.T

X, y_labels = [], []

for root, _, files in os.walk(DATASET_PATH):
    for file in files:
        if file.endswith(".wav"):
            feature = extract_features(os.path.join(root, file))
            X.append(feature)
            # RAVDESS dosya adında duygu kodu var (ör: 03 = happy)
            emotion_code = int(file.split("-")[2])
            emotion_map = {
                1: "neutral", 2: "calm", 3: "happy", 4: "sad",
                5: "angry", 6: "fearful", 7: "disgust", 8: "surprised"
            }
            y_labels.append(emotion_map[emotion_code])

encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y_labels)

X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

X_train = np.array(X_train)
X_test = np.array(X_test)

model = models.Sequential([
    layers.Conv1D(64, kernel_size=3, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2])),
    layers.MaxPooling1D(pool_size=2),
    layers.LSTM(128),
    layers.Dense(64, activation='relu'),
    layers.Dense(len(encoder.classes_), activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=30, batch_size=16, validation_split=0.2)

y_pred = np.argmax(model.predict(X_test), axis=1)
print(classification_report(y_test, y_pred, target_names=encoder.classes_))
