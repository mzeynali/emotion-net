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
        result_texts = (
            f"Emotion: {emotion[0]}",
            f"Valence: {valence:.3f}",
            f"Arousal: {arousal:.3f}"
        )
    else:
        result_text = "No face detected"

    for ti, show_text in enumerate(result_texts):
        cv2.putText(
            frame,
            show_text,
            (50, 50 + ti * 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2,
        )
    cv2.imshow("frame", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        cv2.destroyAllWindows()
        cap.release()
        break
