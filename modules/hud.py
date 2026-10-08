import cv2
import time


class HUD:

    def __init__(self):

        # HUD starts hidden
        self.visible = False

        self.notification = ""
        self.notification_color = (0, 255, 0)

        self.notification_start = 0
        self.notification_duration = 2

    def toggle(self):

        self.visible = not self.visible

    def show_notification(self, text, color=(0, 255, 0)):

        self.notification = text
        self.notification_color = color
        self.notification_start = time.time()

    def draw(
        self,
        frame,
        fps,
        faces,
        hands,
        gesture,
        control_enabled
    ):

        # Do not draw anything when HUD is hidden
        if not self.visible:
            return frame

        h, w = frame.shape[:2]

        overlay = frame.copy()

        cv2.rectangle(
            overlay,
            (15, 15),
            (340, 205),
            (35, 35, 35),
            -1
        )

        frame = cv2.addWeighted(
            overlay,
            0.45,
            frame,
            0.55,
            0
        )

        cv2.rectangle(
            frame,
            (15, 15),
            (340, 205),
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            "MotionOS",
            (28, 45),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (255, 255, 255),
            2
        )

        if control_enabled:

            mode = "CONTROL"
            color = (0, 255, 0)

        else:

            mode = "VIEWER"
            color = (0, 0, 255)

        cv2.circle(
            frame,
            (310, 40),
            7,
            color,
            -1
        )

        cv2.putText(
            frame,
            mode,
            (190, 47),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2
        )

        y = 80

        stats = [
            ("FPS", fps),
            ("Hands", hands),
            ("Faces", faces),
            ("Gesture", gesture)
        ]

        for name, value in stats:

            cv2.putText(
                frame,
                f"{name}:",
                (30, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (200, 200, 200),
                2
            )

            cv2.putText(
                frame,
                str(value),
                (155, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2
            )

            y += 32

        if (
            time.time() - self.notification_start
            < self.notification_duration
        ):

            (tw, th), _ = cv2.getTextSize(
                self.notification,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                2
            )

            x = (w - tw) // 2

            cv2.putText(
                frame,
                self.notification,
                (x, h - 45),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                self.notification_color,
                2
            )

        return frame