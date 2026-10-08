import cv2

# ==========================================
# Hand Skeleton Connections
# ==========================================

HAND_CONNECTIONS = [

    # Thumb
    (0, 1), (1, 2), (2, 3), (3, 4),

    # Index
    (0, 5), (5, 6), (6, 7), (7, 8),

    # Middle
    (0, 9), (9, 10), (10, 11), (11, 12),

    # Ring
    (0, 13), (13, 14), (14, 15), (15, 16),

    # Pinky
    (0, 17), (17, 18), (18, 19), (19, 20),

    # Palm
    (5, 9),
    (9, 13),
    (13, 17)
]


class Renderer:

    # ==========================================
    # Draw Faces
    # ==========================================

    def draw_faces(self, frame, faces):

        for face in faces:

            x, y, w, h = face["bbox"]
            cx, cy = face["center"]
            history = face.get("history", [])

            # Bounding Box
            cv2.rectangle(
                frame,
                (x, y),
                (x + w, y + h),
                (0, 255, 0),
                2
            )

            # Motion Trail
            for i in range(1, len(history)):
                cv2.line(
                    frame,
                    history[i - 1],
                    history[i],
                    (255, 0, 255),
                    2
                )

            # Center
            cv2.circle(
                frame,
                (cx, cy),
                5,
                (0, 0, 255),
                -1
            )

            # Face ID
            cv2.putText(
                frame,
                f"Face #{face.get('id', '?')}",
                (x, y - 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 0),
                2
            )

            # Confidence
            cv2.putText(
                frame,
                f"{face.get('confidence', 0):.2f}",
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 0),
                2
            )

            # Coordinates
            cv2.putText(
                frame,
                f"({cx}, {cy})",
                (x, y + h + 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                2
            )

        return frame

    # ==========================================
    # Draw Hands
    # ==========================================

    def draw_hands(self, frame, hands):

        for hand in hands:

            x, y, w, h = hand["bbox"]
            cx, cy = hand["center"]
            landmarks = hand["landmarks"]

            # ---------------------------------
            # Bounding Box
            # ---------------------------------

            cv2.rectangle(
                frame,
                (x, y),
                (x + w, y + h),
                (255, 150, 0),
                2
            )

            # ---------------------------------
            # Skeleton
            # ---------------------------------

            for start, end in HAND_CONNECTIONS:

                cv2.line(
                    frame,
                    landmarks[start],
                    landmarks[end],
                    (0, 255, 0),
                    2
                )

            # ---------------------------------
            # Thumb ↔ Index
            # ---------------------------------

            thumb = landmarks[4]
            index = landmarks[8]

            cv2.line(
                frame,
                thumb,
                index,
                (0, 0, 255),
                3
            )

            # ---------------------------------
            # Finger Trail
            # ---------------------------------

            history = hand.get("finger_history", [])

            for i in range(1, len(history)):
                cv2.line(
                    frame,
                    history[i - 1],
                    history[i],
                    (255, 0, 255),
                    2
                )

            # ---------------------------------
            # Landmarks
            # ---------------------------------

            for point in landmarks:
                cv2.circle(
                    frame,
                    point,
                    5,
                    (0, 255, 255),
                    -1
                )

            # ---------------------------------
            # Center Point
            # ---------------------------------

            cv2.circle(
                frame,
                (cx, cy),
                6,
                (255, 255, 255),
                -1
            )

            # =================================
            # TEXT LAYOUT
            # =================================

            # PINCH
            if hand.get("pinching", False):

                cv2.putText(
                    frame,
                    "PINCH",
                    (x, y - 105),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.75,
                    (0, 0, 255),
                    2
                )

            # Hand ID
            cv2.putText(
                frame,
                f"Hand #{hand.get('id', '?')}",
                (x, y - 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 0),
                2
            )

            # Gesture
            cv2.putText(
                frame,
                hand.get("gesture", "Unknown"),
                (x, y - 55),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 255, 255),
                2
            )

            # Left / Right
            cv2.putText(
                frame,
                hand.get("handedness", "Unknown"),
                (x, y - 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 150, 0),
                2
            )

            # Confidence
            cv2.putText(
                frame,
                f"{hand.get('confidence', 0):.2f}",
                (x, y + h + 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                2
            )

            # Speed
            if "speed" in hand:

                cv2.putText(
                    frame,
                    f"Speed: {hand['speed']:.1f}",
                    (x, y + h + 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 255),
                    2
                )

            # Pinch Distance
            if "pinch_distance" in hand:

                mid_x = (thumb[0] + index[0]) // 2
                mid_y = (thumb[1] + index[1]) // 2

                cv2.putText(
                    frame,
                    f"{hand['pinch_distance']:.1f}",
                    (mid_x, mid_y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    2
                )

        return frame