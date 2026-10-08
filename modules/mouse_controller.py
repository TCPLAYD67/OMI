import pyautogui


class MouseController:

    def __init__(self, sound):

        # =====================================
        # Sound Manager
        # =====================================

        self.sound = sound

        # =====================================
        # Screen Resolution
        # =====================================

        self.screen_width, self.screen_height = pyautogui.size()

        # =====================================
        # Active Area Margin
        # =====================================

        self.margin = 80

        # =====================================
        # Mirror Settings
        # =====================================

        self.mirror_x = True
        self.mirror_y = False

        # =====================================
        # Cursor State
        # =====================================

        self.prev_x = self.screen_width // 2
        self.prev_y = self.screen_height // 2

        # =====================================
        # Click State
        # =====================================

        self.was_index_pinching = False
        self.was_middle_pinching = False

        pyautogui.FAILSAFE = False
        pyautogui.PAUSE = 0

    # =====================================
    # Cursor Movement
    # =====================================

    def update(self, hand, frame=None):

        # Use the actual camera frame resolution.
        # Keep 640x480 as a safe fallback for compatibility.
        if frame is not None:

            camera_height, camera_width = frame.shape[:2]

        else:

            camera_width = 640
            camera_height = 480

        x, y = hand["landmarks"][8]

        margin_x = min(
            self.margin,
            camera_width // 4
        )

        margin_y = min(
            self.margin,
            camera_height // 4
        )

        # =====================================
        # Clamp To Active Area
        # =====================================

        x = max(
            margin_x,
            min(camera_width - margin_x, x)
        )

        y = max(
            margin_y,
            min(camera_height - margin_y, y)
        )

        active_width = camera_width - (2 * margin_x)
        active_height = camera_height - (2 * margin_y)

        # =====================================
        # Camera -> Screen Mapping
        # =====================================

        screen_x = (
            (x - margin_x) / active_width
        ) * self.screen_width

        screen_y = (
            (y - margin_y) / active_height
        ) * self.screen_height

        # =====================================
        # Mirror
        # =====================================

        if self.mirror_x:

            screen_x = (
                self.screen_width - screen_x
            )

        if self.mirror_y:

            screen_y = (
                self.screen_height - screen_y
            )

        # =====================================
        # Adaptive Cursor Smoothing
        # =====================================

        dx = screen_x - self.prev_x
        dy = screen_y - self.prev_y

        distance = (dx * dx + dy * dy) ** 0.5

        if distance < 15:

            smoothing = 0.40

        elif distance < 60:

            smoothing = 0.28

        elif distance < 150:

            smoothing = 0.18

        else:

            smoothing = 0.10

        self.prev_x += dx * smoothing
        self.prev_y += dy * smoothing

        pyautogui.moveTo(
            int(self.prev_x),
            int(self.prev_y),
            _pause=False
        )

    # =====================================
    # Click Detection
    # =====================================

    def handle_buttons(self, hand):

        left = hand.get(
            "pinching_index",
            False
        )

        right = hand.get(
            "pinching_middle",
            False
        )

        result = None

        # ---------------------------------
        # Left Click
        # ---------------------------------

        if left and not self.was_index_pinching:

            self.sound.click()

            pyautogui.click()

            result = "left"

        # ---------------------------------
        # Right Click
        # ---------------------------------

        elif right and not self.was_middle_pinching:

            self.sound.right_click()

            pyautogui.rightClick()

            result = "right"

        self.was_index_pinching = left
        self.was_middle_pinching = right

        return result
#inverse code
# import pyautogui
#
#
# class MouseController:
#
#     def __init__(self):
#
#         # Screen Resolution
#         self.screen_width, self.screen_height = pyautogui.size()
#
#         # Camera Resolution
#         self.camera_width = 640
#         self.camera_height = 480
#
#         # Active Area
#         self.margin = 80
#
#         # Smoothed Cursor Position
#         self.prev_x = self.screen_width // 2
#         self.prev_y = self.screen_height // 2
#
#         # Smoothing Factor
#         self.smoothing = 0.25
#
#         # Disable failsafe (cursor to top-left)
#         pyautogui.FAILSAFE = False
#
#     def update(self, hand):
#
#         # Index fingertip
#         x, y = hand["landmarks"][8]
#
#         # --------------------------------
#         # Clamp to active area
#         # --------------------------------
#
#         x = max(self.margin,
#                 min(self.camera_width - self.margin, x))
#
#         y = max(self.margin,
#                 min(self.camera_height - self.margin, y))
#
#         # --------------------------------
#         # Convert to screen coordinates
#         # --------------------------------
#
#         screen_x = (
#             (x - self.margin)
#             / (self.camera_width - 2 * self.margin)
#         ) * self.screen_width
#
#         screen_y = (
#             (y - self.margin)
#             / (self.camera_height - 2 * self.margin)
#         ) * self.screen_height
#
#         # --------------------------------
#         # Smooth movement
#         # --------------------------------
#
#         self.prev_x += (screen_x - self.prev_x) * self.smoothing
#         self.prev_y += (screen_y - self.prev_y) * self.smoothing
#
#         # --------------------------------
#         # Move mouse
#         # --------------------------------
#
#         pyautogui.moveTo(
#             self.prev_x,
#             self.prev_y,
#             _pause=False
#         )