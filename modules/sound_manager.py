import winsound


class SoundManager:

    def __init__(self):
        self.enabled = True

    # ==========================================
    # Left Click
    # ==========================================

    def click(self):

        if self.enabled:

            winsound.MessageBeep(
                winsound.MB_OK
            )

    # ==========================================
    # Right Click
    # ==========================================

    def right_click(self):

        if self.enabled:

            winsound.MessageBeep(
                winsound.MB_ICONQUESTION
            )

    # ==========================================
    # Drag Start
    # ==========================================

    def drag_start(self):

        if self.enabled:

            winsound.MessageBeep(
                winsound.MB_ICONASTERISK
            )

    # ==========================================
    # Drag End
    # ==========================================

    def drag_end(self):

        if self.enabled:

            winsound.MessageBeep(
                winsound.MB_ICONEXCLAMATION
            )

    # ==========================================
    # Scroll Start
    # ==========================================

    def scroll_start(self):

        if self.enabled:

            winsound.MessageBeep(
                winsound.MB_ICONQUESTION
            )

    # ==========================================
    # Control Enabled
    # ==========================================

    def enable(self):

        if self.enabled:

            winsound.MessageBeep(
                winsound.MB_ICONASTERISK
            )

    # ==========================================
    # Control Disabled
    # ==========================================

    def disable(self):

        if self.enabled:

            winsound.MessageBeep(
                winsound.MB_ICONHAND
            )