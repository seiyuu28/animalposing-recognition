import tensorflow as tf

import numpy as np

from PIL import Image
 
# --- 設定 ---

model_path = 'animalmodel5.h5'       # または SavedModel フォルダ

image_path = 'usa_hint.jpg'

label_map = ['bard_posing', 'bodybilder_posing', 'cat_posing', 'dog_posing' , 'kirin_posing', 'monk_posing','usagi_posing']  # 学習時のクラス順に合わせる

target_size = (300, 300)  # モデル入力サイズ
 
# --- モデル読み込み ---

model = tf.keras.models.load_model(model_path)
 
# --- 画像読み込み＆前処理 ---

img = Image.open(image_path).convert('RGB')

img = img.resize(target_size)                 # リサイズ

x = np.array(img, dtype=np.float32) / 255.0    # 正規化

x = np.expand_dims(x, axis=0)                  # バッチ次元追加
 
# （もし学習時に別の前処理—for example mean/std での標準化—をしていたらここで同様に適用）
 
# --- 推論 ---

probs = model.predict(x)[0]

pred_idx = np.argmax(probs)

pred_label = label_map[pred_idx]

confidence = probs[pred_idx]
 
# --- 結果出力 ---

print(f"予測ポーズ: {pred_label} (信頼度: {confidence:.2%})")
 