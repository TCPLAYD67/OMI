from pathlib import Path

import cv2
import numpy as np
import insightface

from insightface.utils import face_align


class FaceEmbeddingExtractor:

    def __init__(self):

        model_path = (
            Path.home()
            / ".insightface"
            / "models"
            / "buffalo_l"
            / "w600k_r50.onnx"
        )

        if not model_path.exists():

            raise FileNotFoundError(
                "\nArcFace model not found.\n\n"
                f"Expected:\n{model_path}\n"
            )

        # Load only the ArcFace recognition model.
        self.recognizer = (
            insightface.model_zoo.get_model(
                str(model_path),
                providers=[
                    "CUDAExecutionProvider",
                    "CPUExecutionProvider"
                ]
            )
        )

        self.recognizer.prepare(
            ctx_id=0
        )

        # ------------------------------------------------
        # Stable MediaPipe eye landmarks
        # ------------------------------------------------
        #
        # LEFT EYE:
        # 33  = outer corner
        # 133 = inner corner
        # 159 = upper eyelid
        # 145 = lower eyelid
        #
        # RIGHT EYE:
        # 263 = outer corner
        # 362 = inner corner
        # 386 = upper eyelid
        # 374 = lower eyelid
        #
        # Instead of using iris centers, we calculate
        # the geometric center of the eye region.
        #
        # This makes alignment less sensitive to where
        # the person is looking.

        self.left_eye_indices = [
            33,
            133,
            159,
            145
        ]

        self.right_eye_indices = [
            263,
            362,
            386,
            374
        ]

        # Nose tip
        self.nose_index = 1

        # Mouth corners
        self.left_mouth_index = 61
        self.right_mouth_index = 291

    def _average_points(
        self,
        landmarks,
        indices
    ):

        points = []

        for index in indices:

            x, y = landmarks[index]

            points.append([
                float(x),
                float(y)
            ])

        points = np.asarray(
            points,
            dtype=np.float32
        )

        return np.mean(
            points,
            axis=0
        )

    def get_alignment_points(
        self,
        face
    ):

        landmarks = face.get(
            "landmarks",
            []
        )

        if len(landmarks) < 292:

            return None

        # Stable eye centers.
        left_eye = self._average_points(
            landmarks,
            self.left_eye_indices
        )

        right_eye = self._average_points(
            landmarks,
            self.right_eye_indices
        )

        # Nose.
        nose = np.asarray(
            landmarks[self.nose_index],
            dtype=np.float32
        )

        # Mouth corners.
        left_mouth = np.asarray(
            landmarks[self.left_mouth_index],
            dtype=np.float32
        )

        right_mouth = np.asarray(
            landmarks[self.right_mouth_index],
            dtype=np.float32
        )

        points = np.asarray(
            [
                left_eye,
                right_eye,
                nose,
                left_mouth,
                right_mouth
            ],
            dtype=np.float32
        )

        return points

    def align_face(
        self,
        frame,
        face
    ):

        alignment_points = (
            self.get_alignment_points(
                face
            )
        )

        if alignment_points is None:

            return None

        # ArcFace's standard five-point 112x112
        # alignment.
        aligned = face_align.norm_crop(
            frame,
            landmark=alignment_points,
            image_size=112
        )

        return aligned

    def extract(
        self,
        frame,
        face
    ):

        aligned = self.align_face(
            frame,
            face
        )

        if aligned is None:

            return None

        # Generate ArcFace embedding.
        embedding = (
            self.recognizer.get_feat(
                aligned
            )[0]
        )

        embedding = np.asarray(
            embedding,
            dtype=np.float32
        ).flatten()

        # L2 normalize.
        norm = np.linalg.norm(
            embedding
        )

        if norm <= 1e-12:

            return None

        embedding = (
            embedding / norm
        )

        return embedding

    def get_debug_alignment(
        self,
        frame,
        face
    ):

        """
        Returns a debug image showing the five
        alignment points used by ArcFace.
        """

        alignment_points = (
            self.get_alignment_points(
                face
            )
        )

        if alignment_points is None:

            return frame.copy()

        debug = frame.copy()

        for index, point in enumerate(
            alignment_points
        ):

            x = int(point[0])
            y = int(point[1])

            cv2.circle(
                debug,
                (x, y),
                4,
                (0, 255, 0),
                -1,
                cv2.LINE_AA
            )

            cv2.putText(
                debug,
                str(index + 1),
                (x + 6, y - 6),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 255, 0),
                1,
                cv2.LINE_AA
            )

        return debug

    def close(self):

        self.recognizer = None