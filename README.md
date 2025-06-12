# animalTOYONAKA

This repository contains an example script for collecting pose data using [MediaPipe](https://github.com/google/mediapipe) and training a simple neural network with TensorFlow.

## Requirements

- Python 3
- OpenCV (`opencv-python`)
- MediaPipe
- TensorFlow
- scikit-learn

## Usage

1. Install the above dependencies.
2. Run the script and input a label name for the capture session when prompted. Press `q` to stop capturing frames.

```bash
python3 pose_trainer.py
```

The script saves collected data to `pose_data.npy` and `pose_labels.npy`, then trains a small model and stores it in `pose_model.h5`.
