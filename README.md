# animalposing-recognition

This repository contains an example script for collecting pose data using [MediaPipe](https://github.com/google/mediapipe) and training a simple neural network with TensorFlow.

## Requirements

- Python 3
- [OpenCV](https://pypi.org/project/opencv-python/)
- [MediaPipe](https://pypi.org/project/mediapipe/)
- [TensorFlow](https://www.tensorflow.org/)
- [scikit-learn](https://scikit-learn.org/)

Install dependencies via pip:

```bash
pip install -r requirements.txt
```

## Usage

1. Install the above dependencies.
2. Run the script. You will be prompted for a label name for the capture session. Press `q` to stop capturing frames. After capturing, the model is trained automatically.

```bash
python3 pose_trainer.py
```

The script saves collected data to `pose_data.npy` and `pose_labels.npy`, then trains a small model and stores it in `pose_model.h5`.
