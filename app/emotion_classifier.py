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
    - It returns a list of (emotion_label, box, confidence) per face.

      box = (x1, y1, x2, y2)
      confidence = float in [0.0, 1.0]
    """

    def __init__(
        self,
        det_weights: str = "models/model_face_detection.pt",
        emotion_pt: str = "models/model_emotion_classifier.pt",
        classify_every: int = 3,
        conf: float = 0.5,
    ) -> None:

        self.det_model = YOLO(det_weights)

        self.emotion_model = YOLO(emotion_pt)

        self.class_names = {
            i: name.lower() for i, name in self.emotion_model.names.items()
        }
        print("Loaded emotion classes:", self.class_names)

        self.classify_every = max(1, int(classify_every))
        self.conf = conf

        self.frame_count = 0
        self.last_emotions = []

        self._last_debug_emotion = None
        self._last_debug_window = None

    def _classify_face(self, face_bgr: np.ndarray):
        """
        Classify a cropped face region.

        - Converts to grayscale but keeps original H x W
        - Converts back to 3-channel so YOLO can consume it
        - Shows the cropped grayscale face in a window
        - Returns (emotion_label, confidence)
        """
        if face_bgr is None or face_bgr.size == 0:
            return "unknown", 0.0

        if face_bgr.ndim == 2:
            gray = face_bgr
        elif face_bgr.ndim == 3 and face_bgr.shape[2] == 1:
            gray = face_bgr[..., 0]
        else:
            # Normal case: BGR -> gray
            gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)

        gray_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

        results = self.emotion_model(gray_bgr, imgsz=640, verbose=False)
        if not results:
            return "unknown", 0.0

        r = results[0]
        if r.probs is None:
            return "unknown", 0.0

        pred_idx = int(r.probs.top1)
        # correct YOLO top-1 confidence
        confidence = float(r.probs.top1conf)

        # YOLO’s built-in class name
        raw_name = self.class_names.get(pred_idx, "unknown")

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

        mapped = RAW_TO_PIPE.get(raw_name, "unknown")

        # ---------- cropped grayscale face ----------
        # if mapped not in ("unknown", "..."):
        #     window_name = f"Face – {mapped}"
        #
        #     if self._last_debug_window is not None and window_name != self._last_debug_window:
        #         try:
        #             cv2.destroyWindow(self._last_debug_window)
        #         except cv2.error:
        #             pass
        #
        #     cv2.imshow(window_name, gray)
        #
        #     self._last_debug_emotion = mapped
        #     self._last_debug_window = window_name

        return mapped, confidence

    def process_frame_multi(self, frame_bgr: np.ndarray):
        """
        Multi-face API.
        :param frame_bgr: input frame (BGR).
        :return: list of (emotion_text, box, confidence)
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

        emotions = []

        # ---- Classify every N frames to save FPS ----
        if faces and (self.frame_count % self.classify_every == 0):
            for face_crop, box in faces:
                emo, conf = self._classify_face(face_crop)
                emotions.append((emo, box, conf))
            self.last_emotions = emotions
        else:
            if self.last_emotions and len(self.last_emotions) == len(face_boxes):
                emotions = [
                    (emo, new_box, conf)
                    for (emo, _old_box, conf), new_box in zip(self.last_emotions, face_boxes)
                ]
            else:
                emotions = [("...", box, 0.0) for box in face_boxes]
                self.last_emotions = emotions

        return emotions