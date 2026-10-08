class FingerState:

    def get_states(self, hand):

        lm = hand["landmarks"]

        states = {}

        # Thumb (temporary version)
        states["thumb"] = lm[4][0] > lm[3][0]

        # Index
        states["index"] = lm[8][1] < lm[6][1]

        # Middle
        states["middle"] = lm[12][1] < lm[10][1]

        # Ring
        states["ring"] = lm[16][1] < lm[14][1]

        # Pinky
        states["pinky"] = lm[20][1] < lm[18][1]

        return states