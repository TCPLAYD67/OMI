import cv2
import mediapipe as mp

from mediapipe.tasks.python import vision
from mediapipe.tasks.python import BaseOptions


class HandDetector:

    def __init__(self):

        options = vision.HandLandmarkerOptions(

            base_options=BaseOptions(
                model_asset_path="models/hand_landmarker.task"
            ),

            running_mode=vision.RunningMode.VIDEO,

            num_hands=2,

            min_hand_detection_confidence=0.40,
            min_hand_presence_confidence=0.40,
            min_tracking_confidence=0.40

        )

        self.detector = vision.HandLandmarker.create_from_options(options)

        # Timestamp required for VIDEO mode (milliseconds)
        self.timestamp = 0

    def detect(self, frame):

        # Advance timestamp (~30 FPS)
        self.timestamp += 33

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb
        )

        result = self.detector.detect_for_video(
            mp_image,
            self.timestamp
        )

        hands = []

        h, w, _ = frame.shape

        for hand_index, landmarks in enumerate(result.hand_landmarks):

            points = []

            x_values = []
            y_values = []

            for landmark in landmarks:

                x = int(landmark.x * w)
                y = int(landmark.y * h)

                points.append((x, y))

                x_values.append(x)
                y_values.append(y)

            # ----------------------------------
            # Bounding Box
            # ----------------------------------

            padding = 20

            x_min = max(0, min(x_values) - padding)
            y_min = max(0, min(y_values) - padding)

            x_max = min(w, max(x_values) + padding)
            y_max = min(h, max(y_values) + padding)

            bbox = (
                x_min,
                y_min,
                x_max - x_min,
                y_max - y_min
            )

            center = (
                (x_min + x_max) // 2,
                (y_min + y_max) // 2
            )

            handedness = result.handedness[hand_index][0].category_name

            confidence = result.handedness[hand_index][0].score

            hand = {
                "bbox": bbox,
                "center": center,
                "landmarks": points,
                "handedness": handedness,
                "confidence": confidence
            }

            hands.append(hand)

        return frame, hands