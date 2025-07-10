import numpy as np
import mediapipe as mp
from mediapipe.framework.formats import landmark_pb2

# ─ 画面描画用 ──────────────────────────────
def draw_keypoints_line(result, img):
    if not result.pose_landmarks:
        return
    lst = landmark_pb2.NormalizedLandmarkList()
    for p in result.pose_landmarks[0]:        # 1 人目
        lm = lst.landmark.add()
        lm.x, lm.y, lm.z = p.x, p.y, p.z
    mp.solutions.drawing_utils.draw_landmarks(
        img, lst, mp.solutions.pose.POSE_CONNECTIONS)

# ─ 学習前の正規化 ──────────────────────────
def preprocessing_pose_data(arr_obj):
    """
    arr_obj = ndarray(dtype=object)  [[label, x1,y1,z1,…], …]
    戻り値  = (N,99) float32   0〜1 正規化
    """
    arr = np.array([row[1:] for row in arr_obj], dtype=np.float32)
    xs, ys = arr[:, ::3], arr[:, 1::3]
    mx, my = xs.min(1, keepdims=True), ys.min(1, keepdims=True)
    rngx   = xs.max(1, keepdims=True) - mx + 1e-6
    rngy   = ys.max(1, keepdims=True) - my + 1e-6
    arr[:, ::3]  = (xs - mx) / rngx
    arr[:, 1::3] = (ys - my) / rngy
    return arr
