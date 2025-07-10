import os,cv2,csv,time,numpy as np, mediapipe as mp, urllib.request, tensorflow as tf
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.framework.formats import landmark_pb2
from pose_utils import preprocessing_pose_data

id2label = {int(i):n for i,n in csv.reader(open("models/label_map.csv","r",encoding='utf-8'))}
inter=tf.lite.Interpreter(model_path="models/pose_model.tflite"); inter.allocate_tensors()
in_idx=inter.get_input_details()[0]['index']; out_idx=inter.get_output_details()[0]['index']

MODEL_DIR="models"; TASK=os.path.join(MODEL_DIR,"pose_landmarker.task")
if not os.path.exists(TASK):
    urllib.request.urlretrieve("https://storage.googleapis.com/mediapipe-models/"
        "pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task",TASK)
asset=open(TASK,'rb').read()
landmarker=vision.PoseLandmarker.create_from_options(vision.PoseLandmarkerOptions(
     base_options=python.BaseOptions(model_asset_buffer=asset),running_mode=vision.RunningMode.VIDEO))

cap=cv2.VideoCapture(0)
while cap.isOpened():
    ok,frm=cap.read(); frm=cv2.flip(frm,1); h,w,_=frm.shape
    res=landmarker.detect_for_video(mp.Image(mp.ImageFormat.SRGB,cv2.cvtColor(frm,cv2.COLOR_BGR2RGB)),int(time.time()*1000))
    txt="NoPose"
    # --- 推論ループ内 ---------------------------------
    if res.pose_landmarks:
        lms = res.pose_landmarks[0]          # 1人目 33 点
        row = ["dummy"]                      # ダミーラベル
        for lm in lms:
            row.extend([lm.x * w, lm.y * h, lm.z])   # 数値だけを追加

        X = preprocessing_pose_data(np.array([row], dtype=object)).astype("float32")
        inter.set_tensor(in_idx, X)
        inter.invoke()
        probs = inter.get_tensor(out_idx)[0]
        idx   = int(np.argmax(probs))
        txt   = f"{id2label[idx]}  {probs[idx]:.2f}"

        # ランドマーク描画
        proto = landmark_pb2.NormalizedLandmarkList()
        for lm in lms:
            p = proto.landmark.add()
            p.x, p.y, p.z = lm.x, lm.y, lm.z
        mp.solutions.drawing_utils.draw_landmarks(
            frm, proto, mp.solutions.pose.POSE_CONNECTIONS)
    else:
        txt = "NoPose"

    cv2.putText(frm,txt,(10,40),cv2.FONT_HERSHEY_SIMPLEX,1.2,(0,0,255),3)
    cv2.imshow("Pose",frm)
    if cv2.waitKey(1)&0xFF==27: break
cap.release(); cv2.destroyAllWindows(); landmarker.close()