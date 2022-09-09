# Emotion Net

Landmark, emotion, valence, and arousal prediction.


## Installation
This package works standalone if you want to process some normalized face in the proper way. But best solution is to use [face_tractor]() to extract the face bounding and ther feed it `EmotionNet` class. In this way `preprocess_emonet` function will be called automatically and you will get the normalized face. So we recommend to install `face_tractor` :
```bash
git clone https://github.com/mzeynali/face-tractor.git
cd face_tractor
pip install -e .
```

and then install this package:
```bash
git clone https://github.com/mzeynali/emotion-net.git
cd emotion_net
pip install -e .
```

## Usage
You can use this class in the following way:
```python
import cv2
from face_tractor import FaceTractor

from emotion_net import EmotionNet

checkpoint_path = "checkpoints/emo.onnx"
face_extractor = FaceTractor(normalized_distance=0.6)
emotion_model = EmotionNet(checkpoint_path)
cap = cv2.VideoCapture(0)
while cap.isOpened():
    ret, frame = cap.read()
    faces = face_extractor.normalized_faces(frame)
    if len(faces) > 0:
        face = faces[0]
        emotion, valence, arousal = emotion_model(frame, face.bbox)
```