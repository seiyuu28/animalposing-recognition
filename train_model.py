"""train_model.py
X.npy, y.npy → Keras h5 + TFLite 生成
"""
import numpy as np, tensorflow as tf, os, csv
from tensorflow.keras import layers, models

X = np.load("data/X.npy"); y = np.load("data/y.npy")
num_classes = len(set(y))
Y = tf.keras.utils.to_categorical(y, num_classes)

model = models.Sequential([
    layers.Input((99,)), layers.Dense(128,activation='relu'), layers.Dropout(0.3),
    layers.Dense(64,activation='relu'), layers.Dropout(0.3),
    layers.Dense(num_classes, activation='softmax')])
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X,Y,epochs=30,batch_size=16)
model.save("models/pose_model.h5")
print("Saved pose_model.h5")

# TFLite
conv = tf.lite.TFLiteConverter.from_keras_model(model); tflite = conv.convert()
open("models/pose_model.tflite","wb").write(tflite)
print("Saved pose_model.tflite")

# ラベル表
with open("models/label_map.csv","w",encoding='utf-8') as f:
    for idx,name in sorted({v:k for k,v in enumerate(set(y))}.items()):
        f.write(f"{idx},{name}\n")
print("label_map.csv saved")