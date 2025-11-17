import cv2
import numpy as np
from ultralytics import YOLO

EMOTION_DICT = {
    0: "neutral",
    1: "happiness",
    2: "surprise",
    3: "sadness",
    4: "anger",
    5: "disgust",
    6: "fear",
}


class EmotionDetector:
    """
    Pure emotion-detection module.
    - You give it a frame (BGR).
    - It returns (emotion_label, largest_face_box).

    It runs YOLO every frame, and the FER+ classifier every N frames
    to keep FPS higher.
    """

    def __init__(
        self,
        det_weights: str = "yolov12n-face.pt",
        emotion_onnx: str = "emotion-ferplus-8.onnx",
        classify_every: int = 3,
        conf: float = 0.5,
    ) -> None:
        self.det_model = YOLO(det_weights)
        self.emotion_net = cv2.dnn.readNetFromONNX(emotion_onnx)
        self.classify_every = max(1, int(classify_every))
        self.conf = conf

        self.frame_count = 0
        self.current_emotion = "..."
        self.last_box = None

    def _classify_face(self, face_bgr: np.ndarray) -> str:
        """Run FER+ ONNX on a cropped face and return the label."""
        gray_face = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
        face_64 = cv2.resize(gray_face, (64, 64))

        blob = face_64.astype("float32")
        blob = np.expand_dims(blob, axis=0)  # (1, 64, 64)
        blob = np.expand_dims(blob, axis=0)  # (1, 1, 64, 64)

        self.emotion_net.setInput(blob)
        out = self.emotion_net.forward()  # (1, N)
        pred_idx = int(np.argmax(out[0]))
        return EMOTION_DICT.get(pred_idx, "unknown")

    def process_frame(self, frame_bgr: np.ndarray):
        """
        Main API.
        :param frame_bgr: input frame (already flipped if you want mirror).
        :return: (emotion_text, largest_box)
                 largest_box = (x1, y1, x2, y2) or None
        """
        self.frame_count += 1

        results = self.det_model(frame_bgr, verbose=False, conf=self.conf)

        largest_face = None
        largest_box = None
        largest_area = 0

        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                area = (x2 - x1) * (y2 - y1)
                if area > largest_area:
                    largest_area = area
                    largest_box = (x1, y1, x2, y2)
                    largest_face = frame_bgr[y1:y2, x1:x2]

        # Only run classifier every N frames
        if (
            largest_face is not None
            and largest_face.size > 0
            and (self.frame_count % self.classify_every == 0)
        ):
            self.current_emotion = self._classify_face(largest_face)

        self.last_box = largest_box
        return self.current_emotion, largest_box
