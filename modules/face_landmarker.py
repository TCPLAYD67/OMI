from pathlib import Path

import cv2
import mediapipe as mp


class FaceLandmarker:

    def __init__(self):

        project_root = (
            Path(__file__).resolve().parent.parent
        )

        self.model_path = (
            project_root
            / "models"
            / "face_landmarker.task"
        )

        if not self.model_path.exists():

            raise FileNotFoundError(
                "\nFace Landmarker model not found.\n\n"
                f"Expected:\n{self.model_path}\n\n"
                "Make sure face_landmarker.task is "
                "inside the models folder."
            )

        BaseOptions = mp.tasks.BaseOptions
        VisionRunningMode = (
            mp.tasks.vision.RunningMode
        )

        options = (
            mp.tasks.vision.FaceLandmarkerOptions(
                base_options=BaseOptions(
                    model_asset_path=str(
                        self.model_path
                    )
                ),

                running_mode=VisionRunningMode.VIDEO,

                num_faces=5,

                min_face_detection_confidence=0.40,
                min_face_presence_confidence=0.40,
                min_tracking_confidence=0.40,

                output_face_blendshapes=False,
                output_facial_transformation_matrixes=False
            )
        )

        self.landmarker = (
            mp.tasks.vision.FaceLandmarker
            .create_from_options(
                options
            )
        )

        self.timestamp_ms = 0

        self.connections = (
            mp.tasks.vision.FaceLandmarksConnections
            .FACE_LANDMARKS_TESSELATION
        )

    def detect(self, frame):

        if frame is None:
            return frame, []

        height, width = frame.shape[:2]

        rgb_frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame
        )

        self.timestamp_ms += 1

        result = self.landmarker.detect_for_video(
            mp_image,
            self.timestamp_ms
        )

        faces = []

        for face_landmarks in (
            result.face_landmarks
        ):

            points = []

            min_x = width
            min_y = height

            max_x = 0
            max_y = 0

            for landmark in face_landmarks:

                x = int(
                    landmark.x * width
                )

                y = int(
                    landmark.y * height
                )

                x = max(
                    0,
                    min(width - 1, x)
                )

                y = max(
                    0,
                    min(height - 1, y)
                )

                points.append(
                    (x, y)
                )

                min_x = min(
                    min_x,
                    x
                )

                min_y = min(
                    min_y,
                    y
                )

                max_x = max(
                    max_x,
                    x
                )

                max_y = max(
                    max_y,
                    y
                )

            if not points:
                continue

            center = (
                (min_x + max_x) // 2,
                (min_y + max_y) // 2
            )

            faces.append({

                "landmarks": points,

                "bbox": (
                    min_x,
                    min_y,
                    max_x,
                    max_y
                ),

                "center": center

            })

        return frame, faces

    def draw(self, frame, faces):

        if not faces:
            return frame

        for face in faces:

            landmarks = face.get(
                "landmarks",
                []
            )

            if not landmarks:
                continue

            # -------------------------
            # Face mesh
            # -------------------------

            for connection in self.connections:

                start = connection.start
                end = connection.end

                if (
                    start >= len(landmarks)
                    or end >= len(landmarks)
                ):
                    continue

                p1 = landmarks[start]
                p2 = landmarks[end]

                cv2.line(
                    frame,
                    p1,
                    p2,
                    (120, 190, 220),
                    1,
                    cv2.LINE_AA
                )

            # -------------------------
            # Landmark points
            # -------------------------

            for point in landmarks:

                cv2.circle(
                    frame,
                    point,
                    1,
                    (150, 210, 235),
                    -1,
                    cv2.LINE_AA
                )

            # -------------------------
            # Identity label
            # -------------------------

            name = face.get(
                "identity",
                "Unknown"
            )

            if name == "Unknown":

                label = "Unknown"

                label_color = (
                    0,
                    165,
                    255
                )

            else:

                label = name

                label_color = (
                    0,
                    255,
                    0
                )

            min_x, min_y, _, _ = face[
                "bbox"
            ]

            label_y = max(
                25,
                min_y - 10
            )

            cv2.putText(
                frame,
                label,
                (
                    min_x,
                    label_y
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                label_color,
                2,
                cv2.LINE_AA
            )

        return frame

    def close(self):

        if self.landmarker is not None:

            self.landmarker.close()

            self.landmarker = None