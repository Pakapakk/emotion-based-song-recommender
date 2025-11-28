import cv2
import numpy as np
from ultralytics import YOLO

# EMOTION_DICT = {
#     0: "neutral",
#     1: "happiness",
#     2: "surprise",
#     3: "sadness",
#     4: "anger",
#     5: "disgust",
#     6: "fear",
# }

EMOTION_DICT = {
    0: "angry",
    1: "contempt",
    2: "disgust",
    3: "fear",
    4: "happy",
    5: "neutral",
    6: "sad",
    7: "surprise",
}


class EmotionDetector:
    """
    Emotion-detection module (multi-face version).
    - You give it a frame (BGR).
    - It returns a list of (emotion_label, box) per face.

    box = (x1, y1, x2, y2)
    """

    def __init__(
        self,
        det_weights: str = "../models/model_face_detection.pt",
        emotion_pt: str = "../models/model_emotion_classifier.pt",
        classify_every: int = 3,
        conf: float = 0.5,
    ) -> None:
        self.det_model = YOLO(det_weights)
        self.emotion_model = YOLO(emotion_pt)
        self.class_names = {i: name.lower() for i, name in self.emotion_model.names.items()}
        print("Loaded emotion classes:", self.class_names)

        self.classify_every = max(1, int(classify_every))
        self.conf = conf

        self.frame_count = 0
        self.last_emotions = []

    def _classify_face(self, face_bgr: np.ndarray) -> str:
        if face_bgr.ndim == 2:
            face_bgr = cv2.cvtColor(face_bgr, cv2.COLOR_GRAY2BGR)
        elif face_bgr.shape[2] == 1:
            face_bgr = cv2.cvtColor(face_bgr, cv2.COLOR_GRAY2BGR)

        results = self.emotion_model(face_bgr, imgsz=640, verbose=False)
        if not results:
            return "unknown"

        r = results[0]
        if r.probs is None:
            return "unknown"

        pred_idx = int(r.probs.top1)

        # YOLO’s built-in class names
        raw_name = self.class_names.get(pred_idx, "unknown")

        # Normalize to your EmotionRecommender emotion keys
        RAW_TO_PIPE = {
            "angry": "angry",
            "contempt": "contempt",
            "disgust": "disgust",
            "fear": "fear",
            "happy": "happy",
            "neutral": "neutral",
            "sad": "sad",
            "suprise": "surprise",  # fix typo
            "surprise": "surprise",
        }

        return RAW_TO_PIPE.get(raw_name, "unknown")

    def process_frame_multi(self, frame_bgr: np.ndarray):
        """
        Multi-face API.
        :param frame_bgr: input frame (BGR).
        :return: list of (emotion_text, box)
                 box = (x1, y1, x2, y2)
        """
        self.frame_count += 1

        # ---- YOLO face detection ----
        results = self.det_model(frame_bgr, verbose=False, conf=self.conf)

        faces = []
        face_boxes = []

        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                # clip to frame
                h, w = frame_bgr.shape[:2]
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)

                if x2 <= x1 or y2 <= y1:
                    continue

                face_crop = frame_bgr[y1:y2, x1:x2]
                if face_crop.size == 0:
                    continue

                faces.append((face_crop, (x1, y1, x2, y2)))
                face_boxes.append((x1, y1, x2, y2))

        # ---- Classify every N frames to save FPS ----
        emotions = []
        if faces and (self.frame_count % self.classify_every == 0):
            for face_crop, box in faces:
                emo = self._classify_face(face_crop)
                emotions.append((emo, box))
            self.last_emotions = emotions
        else:
            if self.last_emotions and len(self.last_emotions) == len(face_boxes):
                emotions = [
                    (emo, box) for (emo, _old_box), box in zip(self.last_emotions, face_boxes)
                ]
            else:
                emotions = [("...", box) for box in face_boxes]
                self.last_emotions = emotions

        return emotions