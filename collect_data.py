"""collect_data.py
カメラから 33 ポーズランドマークを取り出し、ラベル付きで保存。
操作: n=ラベル設定  c=キャプチャ  q=終了
"""
import os, cv2, pickle, urllib.request, mediapipe as mp, pygame
from time import time
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.framework.formats import landmark_pb2

# ─ モデル (PoseLandmarker) をバイト列でロード ─
MODEL_DIR = "models"; os.makedirs(MODEL_DIR, exist_ok=True)
TASK = os.path.join(MODEL_DIR, "pose_landmarker.task")
URL = ("https://storage.googleapis.com/mediapipe-models/"
       "pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task")
if not os.path.exists(TASK):
    urllib.request.urlretrieve(URL, TASK)
asset = open(TASK, "rb").read()

options = vision.PoseLandmarkerOptions(
    base_options=python.BaseOptions(model_asset_buffer=asset),
    running_mode=vision.RunningMode.VIDEO)
landmarker = vision.PoseLandmarker.create_from_options(options)

cap = cv2.VideoCapture(0)
pygame.init(); clock = pygame.time.Clock()
label, samples = None, []
print("n:新ラベル  c:キャプチャ  q:終了")
while cap.isOpened():
    ok, img = cap.read()
    if not ok:
        break
    img = cv2.flip(img, 1)
    h, w, _ = img.shape

    # ─ Pose 推論 ─
    res = landmarker.detect_for_video(
        mp.Image(mp.ImageFormat.SRGB, cv2.cvtColor(img, cv2.COLOR_BGR2RGB)),
        int(time() * 1000)
    )

    lms = None                    # ← 毎フレーム初期化
    if res.pose_landmarks:
        lms = res.pose_landmarks[0]        # 1 人目
        proto = landmark_pb2.NormalizedLandmarkList()
        for lm_src in lms:
            lm = proto.landmark.add()
            lm.x, lm.y, lm.z = lm_src.x, lm_src.y, lm_src.z
        mp.solutions.drawing_utils.draw_landmarks(
            img, proto, mp.solutions.pose.POSE_CONNECTIONS
        )

    # ─ 画面表示 ─
    cv2.putText(img, f"Label: {label}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("Collect", img)

    # ─ キー入力 ─
    k = cv2.waitKey(1) & 0xFF     # ★ ここで k を取得
    if k == ord('n'):
        label = input("ラベル (dog/monkey/…): ").strip()
    elif k == ord('c') and label and lms is not None:
        row = [label]
        for lm_src in lms:
            row.extend([lm_src.x, lm_src.y, lm_src.z])
        samples.append(row)
        print("Samples:", len(samples))
    elif k == ord('q'):
        break

    clock.tick(30)

# ────────────────────────────────────────────────
# 3) データ保存
os.makedirs("data", exist_ok=True)
with open("data/pose_data.bin", "wb") as f:
    pickle.dump(samples, f)
print(f"Saved → data/pose_data.bin  ({len(samples)} samples)")