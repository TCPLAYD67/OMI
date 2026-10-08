import math


class FaceFeatureExtractor:

    def __init__(self):

        # MediaPipe Face Mesh landmark indices.
        #
        # 33  = left eye outer region
        # 263 = right eye outer region
        #
        # These are used as the reference axis so the face
        # representation is less affected by scale and roll.

        self.left_eye_reference = 33
        self.right_eye_reference = 263

    def extract(self, face):

        landmarks = face.get("landmarks", [])

        if len(landmarks) < 264:
            return None

        left_eye = landmarks[
            self.left_eye_reference
        ]

        right_eye = landmarks[
            self.right_eye_reference
        ]

        # Midpoint between the two reference points.
        center_x = (
            left_eye[0] + right_eye[0]
        ) / 2.0

        center_y = (
            left_eye[1] + right_eye[1]
        ) / 2.0

        # Distance between the two reference points.
        dx = (
            right_eye[0] - left_eye[0]
        )

        dy = (
            right_eye[1] - left_eye[1]
        )

        eye_distance = math.hypot(
            dx,
            dy
        )

        if eye_distance < 1:
            return None

        # Angle of the eye-to-eye line.
        angle = math.atan2(
            dy,
            dx
        )

        cos_a = math.cos(-angle)
        sin_a = math.sin(-angle)

        features = []

        for x, y in landmarks:

            # Move the landmark relative to the
            # center between the eyes.
            px = x - center_x
            py = y - center_y

            # Rotate the face so the eye line becomes horizontal.
            rotated_x = (
                px * cos_a
                - py * sin_a
            )

            rotated_y = (
                px * sin_a
                + py * cos_a
            )

            # Normalize according to eye distance.
            normalized_x = (
                rotated_x / eye_distance
            )

            normalized_y = (
                rotated_y / eye_distance
            )

            features.append(
                round(normalized_x, 6)
            )

            features.append(
                round(normalized_y, 6)
            )

        return features

    def describe(self, face):

        features = self.extract(face)

        if features is None:
            return None

        return {
            "landmark_count": len(
                face["landmarks"]
            ),
            "feature_count": len(features),
            "features": features
        }