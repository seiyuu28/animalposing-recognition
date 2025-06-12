import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical


def capture_pose(label, max_frames=200):
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose()
    cap = cv2.VideoCapture(0)

    data = []
    count = 0
    print("Press 'q' to stop capturing")
    while cap.isOpened() and count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image)
        if results.pose_landmarks:
            row = []
            for lm in results.pose_landmarks.landmark:
                row.extend([lm.x, lm.y, lm.z])
            data.append(row)
            count += 1
        cv2.imshow('Pose', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
    labels = [label] * len(data)
    return np.array(data), np.array(labels)


def train_model(X, y):
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    y_cat = to_categorical(y_enc)

    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(X.shape[1],)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(y_cat.shape[1], activation='softmax')
    ])
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    model.save('pose_model.h5')

    with open('label_map.csv', 'w') as f:
        for idx, name in enumerate(le.classes_):
            f.write(f'{idx},{name}\n')

    print('Model saved to pose_model.h5')


def main():
    label = input('Enter label for this capture session: ')
    X, y = capture_pose(label)
    np.save('pose_data.npy', X)
    np.save('pose_labels.npy', y)
    print('Data saved. Starting training...')
    train_model(X, y)


if __name__ == '__main__':
    main()
