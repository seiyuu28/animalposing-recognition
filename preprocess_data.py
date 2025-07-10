"""preprocess_data.py
生データ (pickle) → 正規化済み numpy (X.npy, y.npy)
"""
import os, pickle, numpy as np
from pose_utils import preprocessing_pose_data

raw = pickle.load(open("data/pose_data.bin","rb"))
print("[DEBUG] samples:", len(raw))       # ← ここ
if not raw:
    raise ValueError("pose_data.bin が空です。 collect_data.py でサンプルを取得してください")
arr = np.array(raw, dtype=object)
labels = arr[:,0]
X = preprocessing_pose_data(arr)
label2id = {name:i for i,name in enumerate(sorted(set(labels)))}
y = np.array([label2id[l] for l in labels], dtype=np.int32)

np.save("data/X.npy", X)
np.save("data/y.npy", y)
print("X/y saved", X.shape, y.shape)
