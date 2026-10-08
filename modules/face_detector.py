import cv2
import mediapipe as mp

from mediapipe.tasks.python import vision
from mediapipe.tasks.python import BaseOptions


class FaceDetector:

    def __init__(self):

        options = vision.FaceDetectorOptions(
            base_options=BaseOptions(
                model_asset_path="models/blaze_face_short_range.tflite"
            )
        )

        self.detector = vision.FaceDetector.create_from_options(options)

    def detect(self, frame):

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb
        )

        result = self.detector.detect(mp_image)

        faces = []

        if result.detections:

            for detection in result.detections:

                bbox = detection.bounding_box

                x = bbox.origin_x
                y = bbox.origin_y
                w = bbox.width
                h = bbox.height

                confidence = detection.categories[0].score

                cx = x + w // 2
                cy = y + h // 2

                face = {
                    "bbox": (x, y, w, h),
                    "center": (cx, cy),
                    "confidence": confidence
                }

                faces.append(face)

        return frame, faces