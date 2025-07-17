from flask import Flask, request, jsonify, send_from_directory
import base64
import numpy as np
import tensorflow as tf
from PIL import Image
import io
import cv2
import mediapipe as mp

app = Flask(__name__)

# モデルとラベルをロード
model = tf.keras.models.load_model('pose_model.h5')
class_names = ['bard_posing', 'bodybilder_posing', 'cat_posing', 'dog_posing',
               'kirin_posing', 'monk_posing', 'usagi_posing']

# ポーズ特徴量（99次元）を抽出する関数
def extract_pose_vector(image_np):
    mp_pose = mp.solutions.pose
    with mp_pose.Pose(static_image_mode=True) as pose:
        results = pose.process(image_np)
        if not results.pose_landmarks:
            return None
        keypoints = []
        for lm in results.pose_landmarks.landmark:
            keypoints.extend([lm.x, lm.y, lm.z])
        return np.array(keypoints)

# 最初のページ
@app.route('/')
def index():
    return send_from_directory('.', 'toyotyu.html')

# 推論API
@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        if not data or 'image' not in data:
            return jsonify({'error': '画像データが送信されていません'}), 400

        # base64 → PIL画像
        image_data = data['image'].split(',')[1]
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')

        # PIL → numpy → OpenCV (RGB)
        image_np = np.array(image)
        image_cv = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

        # ポーズベクトル抽出
        pose_vec = extract_pose_vector(image_cv)
        if pose_vec is None:
            return jsonify({'error': 'ポーズが検出できませんでした'}), 400

        pose_vec = pose_vec.reshape(1, -1)  # (1, 99)

        # 推論実行
        predictions = model.predict(pose_vec)[0]

        # 上位スコア順にソート
        results = sorted(
            zip(class_names, predictions),
            key=lambda x: x[1],
            reverse=True
        )

        # JSON形式で返す
        return jsonify({
            'results': [
                {'label': label, 'score': float(score)} for label, score in results
            ]
        })

    except Exception as e:
        print('サーバーエラー:', e)
        return jsonify({'error': str(e)}), 500

# 静的ファイルのルーティング（HTML, JS など）
@app.route('/<path:path>')
def static_file(path):
    return send_from_directory('.', path)

if __name__ == '__main__':
    app.run(debug=True)
