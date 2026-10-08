import json
import sqlite3
from pathlib import Path

import numpy as np


class FaceEmbeddingRecognizer:

    def __init__(
        self,
        database,
        match_threshold=0.67,
        hold_threshold=0.62,
        margin_threshold=0.03,
        required_confirmations=2,
        max_unknown_streak=6,
        top_k=7,
    ):
        self.database = database

        self.match_threshold = float(match_threshold)
        self.hold_threshold = float(hold_threshold)
        self.margin_threshold = float(margin_threshold)

        self.required_confirmations = max(
            1,
            int(required_confirmations)
        )

        self.max_unknown_streak = max(
            1,
            int(max_unknown_streak)
        )

        self.top_k = max(
            3,
            int(top_k)
        )

        self.people = {}
        self.state = {}

        self.reload()

    # ==========================================================
    # PUBLIC API
    # ==========================================================

    def reset(self):
        self.state.clear()
        self.reload()

    def cleanup(self, active_face_keys):
        active = set(active_face_keys)

        stale = [
            key
            for key in self.state
            if key not in active
        ]

        for key in stale:
            del self.state[key]

    def recognize(
        self,
        embedding,
        face_key="single_face",
        face=None
    ):
        """
        Recognize a face using:
        - all enrollment embeddings
        - top-k consistency
        - pose-aware matching
        - identity margin
        - temporal confirmation

        `face` is optional for compatibility.
        Pose-aware matching activates when a FaceLandmarker
        face dictionary containing landmarks is supplied.
        """

        query = self._normalize_embedding(
            embedding
        )

        if query is None:
            return self._unknown_result(
                face_key
            )

        if not self.people:
            return self._unknown_result(
                face_key
            )

        live_pose = self._estimate_pose(
            face
        )

        candidates = []

        # ======================================================
        # PERSON LOOP
        # ======================================================

        for person in self.people.values():

            embeddings = person["embeddings"]

            if embeddings.shape[0] == 0:
                continue

            similarities = np.dot(
                embeddings,
                query
            )

            similarities = np.asarray(
                similarities,
                dtype=np.float32
            )

            if similarities.size == 0:
                continue

            # ==================================================
            # RAW ARCFace MATCH
            # ==================================================

            raw_top1_index = int(
                np.argmax(similarities)
            )

            raw_top1 = float(
                similarities[raw_top1_index]
            )

            # ==================================================
            # POSE-AWARE MATCHING
            # ==================================================

            pose_bonus = 0.0
            pose_match_used = False
            pose_distance = None

            effective_similarities = similarities.copy()

            stored_pose = person.get(
                "pose"
            )

            if (
                live_pose is not None
                and stored_pose is not None
            ):

                pose_count = min(
                    similarities.shape[0],
                    stored_pose.shape[0]
                )

                if pose_count > 0:

                    stored = stored_pose[
                        :pose_count
                    ]

                    live = np.asarray(
                        live_pose,
                        dtype=np.float32
                    )

                    distances = (
                        self._pose_distance(
                            stored,
                            live
                        )
                    )

                    # Convert pose distance into a
                    # 0..1 similarity.
                    pose_similarities = np.exp(
                        -0.5 * (
                            distances / 1.0
                        ) ** 2
                    )

                    # Very small adjustment so ArcFace remains
                    # the dominant signal.
                    pose_adjustment = (
                        0.02
                        * pose_similarities
                    )

                    effective_similarities[
                        :pose_count
                    ] += pose_adjustment

                    best_pose_index = int(
                        np.argmax(
                            effective_similarities[
                                :pose_count
                            ]
                        )
                    )

                    pose_distance = float(
                        distances[
                            best_pose_index
                        ]
                    )

                    pose_bonus = float(
                        pose_adjustment[
                            best_pose_index
                        ]
                    )

                    pose_match_used = True

            # ==================================================
            # SELECT BEST SAMPLE
            # ==================================================

            best_index = int(
                np.argmax(
                    effective_similarities
                )
            )

            best_raw_similarity = float(
                similarities[best_index]
            )

            best_effective_similarity = float(
                effective_similarities[best_index]
            )

            # Never allow pose logic to turn a weak ArcFace
            # match into a strong one.
            score = min(
                1.0,
                best_raw_similarity
                + min(
                    0.02,
                    max(
                        0.0,
                        best_effective_similarity
                        - best_raw_similarity
                    )
                )
            )

            # ==================================================
            # TOP-K RAW CONSISTENCY
            # ==================================================

            k = min(
                self.top_k,
                similarities.size
            )

            top_k_values = np.sort(
                similarities
            )[-k:][::-1]

            top_k_mean = float(
                np.mean(top_k_values)
            )

            # ==================================================
            # SUPPORT
            # ==================================================

            support_cutoff = max(
                0.60,
                best_raw_similarity - 0.08
            )

            support_count = int(
                np.sum(
                    similarities >= support_cutoff
                )
            )

            candidates.append(
                {
                    "id": person["id"],
                    "name": person["name"],

                    # Main recognition score
                    "score": float(score),

                    # Raw ArcFace score
                    "raw_similarity": (
                        best_raw_similarity
                    ),

                    # Best score after tiny pose adjustment
                    "effective_similarity": (
                        best_effective_similarity
                    ),

                    "topk_mean": top_k_mean,

                    "support_count": support_count,

                    "sample_count": int(
                        embeddings.shape[0]
                    ),

                    "pose_bonus": pose_bonus,

                    "pose_distance": pose_distance,

                    "pose_used": pose_match_used,
                }
            )

        if not candidates:
            return self._unknown_result(
                face_key
            )

        # ======================================================
        # SORT CANDIDATES
        # ======================================================

        candidates.sort(
            key=lambda item: (
                item["score"],
                item["raw_similarity"],
                item["topk_mean"],
                item["support_count"],
            ),
            reverse=True
        )

        best = candidates[0]

        if len(candidates) > 1:

            second = candidates[1]

            second_score = float(
                second["score"]
            )

        else:

            second = None
            second_score = 0.0

        margin = float(
            best["score"]
            - second_score
        )

        # ======================================================
        # MATCH DECISION
        # ======================================================

        strong_match = (
            best["raw_similarity"]
            >= self.match_threshold
            and (
                second is None
                or margin
                >= self.margin_threshold
            )
        )

        hold_match = (
            best["raw_similarity"]
            >= self.hold_threshold
            and (
                second is None
                or margin
                >= self.margin_threshold * 0.50
            )
        )

        result = self._update_temporal_state(
            face_key=face_key,
            best=best,
            margin=margin,
            strong_match=strong_match,
            hold_match=hold_match,
        )

        result.update(
            {
                "top_similarity": (
                    best["raw_similarity"]
                ),

                "topk_similarity": (
                    best["topk_mean"]
                ),

                "support_count": (
                    best["support_count"]
                ),

                "sample_count": (
                    best["sample_count"]
                ),

                "second_best": (
                    second["name"]
                    if second is not None
                    else None
                ),

                "second_similarity": (
                    second_score
                ),

                "pose_used": (
                    best["pose_used"]
                ),

                "pose_distance": (
                    best["pose_distance"]
                ),

                "pose_bonus": (
                    best["pose_bonus"]
                ),
            }
        )

        if live_pose is not None:
            result["live_pose"] = {
                "yaw": float(live_pose[0]),
                "pitch": float(live_pose[1]),
                "roll": float(live_pose[2]),
            }
        else:
            result["live_pose"] = None

        return result

    # ==========================================================
    # TEMPORAL STABILITY
    # ==========================================================

    def _update_temporal_state(
        self,
        face_key,
        best,
        margin,
        strong_match,
        hold_match,
    ):
        candidate_name = best["name"]
        score = float(
            best["raw_similarity"]
        )

        state = self.state.setdefault(
            face_key,
            {
                "candidate": None,
                "candidate_count": 0,
                "identity": "Unknown",
                "unknown_streak": 0,
                "similarity": None,
                "margin": None,
            }
        )

        # ======================================================
        # STRONG MATCH
        # ======================================================

        if strong_match:

            state["unknown_streak"] = 0

            if (
                state["candidate"]
                == candidate_name
            ):

                state["candidate_count"] += 1

            else:

                state["candidate"] = (
                    candidate_name
                )

                state["candidate_count"] = 1

            # --------------------------------------------------
            # Confirm
            # --------------------------------------------------

            if (
                state["candidate_count"]
                >= self.required_confirmations
            ):

                state["identity"] = (
                    candidate_name
                )

                state["similarity"] = score
                state["margin"] = float(
                    margin
                )

                return {
                    "name": candidate_name,
                    "similarity": score,
                    "margin": float(margin),
                    "status": "recognized",
                }

            # --------------------------------------------------
            # Still confirming
            # --------------------------------------------------

            if (
                state["identity"]
                != "Unknown"
            ):

                state["similarity"] = score
                state["margin"] = float(
                    margin
                )

                return {
                    "name": state["identity"],
                    "similarity": score,
                    "margin": float(margin),
                    "status": "holding",
                }

            return {
                "name": "Unknown",
                "similarity": score,
                "margin": float(margin),
                "status": "confirming",
            }

        # ======================================================
        # BORDERLINE / HOLD
        # ======================================================

        if (
            hold_match
            and state["identity"]
            != "Unknown"
        ):

            state["unknown_streak"] += 1

            if (
                state["unknown_streak"]
                <= self.max_unknown_streak
            ):

                state["similarity"] = score
                state["margin"] = float(
                    margin
                )

                return {
                    "name": state["identity"],
                    "similarity": score,
                    "margin": float(margin),
                    "status": "holding",
                }

        # ======================================================
        # UNKNOWN
        # ======================================================

        state["unknown_streak"] += 1

        state["candidate"] = None
        state["candidate_count"] = 0

        if (
            state["unknown_streak"]
            >= self.max_unknown_streak
        ):

            state["identity"] = "Unknown"

        state["similarity"] = score
        state["margin"] = float(
            margin
        )

        return {
            "name": state["identity"],
            "similarity": score,
            "margin": float(margin),
            "status": (
                "holding"
                if state["identity"]
                != "Unknown"
                else "unknown"
            ),
        }

    # ==========================================================
    # DATABASE LOADING
    # ==========================================================

    def reload(self):
        self.people.clear()

        db_path = self._resolve_db_path()

        if (
            db_path is None
            or not db_path.exists()
        ):

            print(
                "[Omi] Recognition database not found."
            )

            return

        try:

            connection = sqlite3.connect(
                str(db_path)
            )

            try:

                cursor = connection.cursor()

                cursor.execute(
                    """
                    SELECT id, name
                    FROM people
                    ORDER BY id
                    """
                )

                rows = cursor.fetchall()

                for person_id, name in rows:

                    data = (
                        self._load_person(
                            cursor,
                            person_id,
                            name
                        )
                    )

                    if data is None:
                        continue

                    self.people[
                        int(person_id)
                    ] = data

            finally:

                connection.close()

        except Exception as error:

            print(
                "[Omi] Recognition database load failed:"
            )

            print(error)

            return

        total_samples = sum(
            person["embeddings"].shape[0]
            for person in self.people.values()
        )

        print(
            "[Omi] Recognition database loaded: "
            f"{len(self.people)} people / "
            f"{total_samples} samples"
        )

    def _load_person(
        self,
        cursor,
        person_id,
        name
    ):
        # ======================================================
        # PREFER 100-SAMPLE FEATURE STORE
        # ======================================================

        feature_store = (
            self._load_feature_store(
                person_id,
                name
            )
        )

        if feature_store is not None:

            return {
                "id": int(person_id),
                "name": str(name),
                "embeddings": feature_store[
                    "embeddings"
                ],
                "pose": feature_store[
                    "pose"
                ],
            }

        # ======================================================
        # SQLITE FALLBACK
        # ======================================================

        cursor.execute(
            """
            SELECT embedding
            FROM face_samples
            WHERE person_id = ?
            ORDER BY id
            """,
            (person_id,)
        )

        rows = cursor.fetchall()

        embeddings = []

        for row in rows:

            if not row:
                continue

            vector = (
                self._decode_embedding(
                    row[0]
                )
            )

            if vector is not None:
                embeddings.append(vector)

        if not embeddings:
            return None

        matrix = np.asarray(
            embeddings,
            dtype=np.float32
        )

        matrix = self._normalize_matrix(
            matrix
        )

        if matrix.shape[0] == 0:
            return None

        return {
            "id": int(person_id),
            "name": str(name),
            "embeddings": matrix,
            "pose": None,
        }

    # ==========================================================
    # FEATURE STORE
    # ==========================================================

    def _load_feature_store(
        self,
        person_id,
        name
    ):
        people_directory = getattr(
            self.database,
            "people_directory",
            None
        )

        if people_directory is None:

            data_directory = getattr(
                self.database,
                "data_directory",
                None
            )

            if data_directory is None:
                return None

            people_directory = (
                Path(data_directory)
                / "people"
            )

        people_directory = Path(
            people_directory
        )

        safe_name = self._safe_name(
            name
        )

        person_directory = (
            people_directory
            / (
                f"person_{int(person_id):04d}_"
                f"{safe_name}"
            )
        )

        feature_path = (
            person_directory
            / "registration_data"
            / "feature_store.npz"
        )

        if not feature_path.exists():
            return None

        try:

            with np.load(
                str(feature_path),
                allow_pickle=False
            ) as data:

                if "embeddings" not in data:
                    return None

                embeddings = np.asarray(
                    data["embeddings"],
                    dtype=np.float32
                )

                if (
                    embeddings.ndim != 2
                    or embeddings.shape[1] != 512
                ):
                    return None

                embeddings = (
                    self._normalize_matrix(
                        embeddings
                    )
                )

                if embeddings.shape[0] == 0:
                    return None

                pose = self._extract_pose_arrays(
                    data,
                    embeddings.shape[0]
                )

                return {
                    "embeddings": embeddings,
                    "pose": pose,
                }

        except Exception as error:

            print(
                f"[Omi] Feature store load failed "
                f"for '{name}': {error}"
            )

            return None

    def _extract_pose_arrays(
        self,
        data,
        expected_count
    ):
        yaw = self._find_array(
            data,
            (
                "yaw",
                "pose_yaw",
                "yaws",
            )
        )

        pitch = self._find_array(
            data,
            (
                "pitch",
                "pose_pitch",
                "pitches",
            )
        )

        roll = self._find_array(
            data,
            (
                "roll",
                "pose_roll",
                "rolls",
            )
        )

        if (
            yaw is None
            or pitch is None
            or roll is None
        ):

            return None

        yaw = yaw.reshape(-1)
        pitch = pitch.reshape(-1)
        roll = roll.reshape(-1)

        count = min(
            expected_count,
            yaw.shape[0],
            pitch.shape[0],
            roll.shape[0]
        )

        if count <= 0:
            return None

        pose = np.column_stack(
            (
                yaw[:count],
                pitch[:count],
                roll[:count],
            )
        ).astype(
            np.float32
        )

        finite_rows = np.all(
            np.isfinite(pose),
            axis=1
        )

        pose = pose[
            finite_rows
        ]

        if pose.shape[0] == 0:
            return None

        # If the pose array has fewer valid rows than embeddings,
        # pose matching must be disabled rather than misaligning
        # samples.
        if pose.shape[0] != expected_count:
            return None

        return pose

    def _find_array(
        self,
        data,
        names
    ):
        for name in names:

            if name in data:

                try:

                    return np.asarray(
                        data[name],
                        dtype=np.float32
                    )

                except Exception:
                    return None

        return None

    # ==========================================================
    # LIVE POSE
    # ==========================================================

    def _estimate_pose(
        self,
        face
    ):
        if not isinstance(
            face,
            dict
        ):
            return None

        landmarks = face.get(
            "landmarks"
        )

        if landmarks is None:
            return None

        points = self._landmarks_to_array(
            landmarks
        )

        if points is None:
            return None

        if points.shape[0] <= 291:
            return None

        try:

            left_eye = np.mean(
                points[
                    [33, 133]
                ],
                axis=0
            )

            right_eye = np.mean(
                points[
                    [263, 362]
                ],
                axis=0
            )

            nose = points[1]

            mouth_left = points[61]
            mouth_right = points[291]

        except Exception:

            return None

        eye_center = (
            left_eye
            + right_eye
        ) / 2.0

        eye_vector = (
            right_eye
            - left_eye
        )

        eye_distance = float(
            np.linalg.norm(
                eye_vector[:2]
            )
        )

        if eye_distance < 1e-6:
            return None

        # ======================================================
        # ROLL
        # ======================================================

        roll = np.degrees(
            np.arctan2(
                eye_vector[1],
                eye_vector[0]
            )
        )

        # ======================================================
        # YAW
        # ======================================================

        nose_offset_x = (
            nose[0]
            - eye_center[0]
        )

        yaw_ratio = (
            nose_offset_x
            / eye_distance
        )

        yaw = (
            yaw_ratio
            * 55.0
        )

        # ======================================================
        # PITCH
        # ======================================================

        mouth_center_y = (
            mouth_left[1]
            + mouth_right[1]
        ) / 2.0

        vertical_span = abs(
            mouth_center_y
            - eye_center[1]
        )

        if vertical_span < 1e-6:
            vertical_span = 1.0

        nose_position = (
            nose[1]
            - eye_center[1]
        ) / vertical_span

        # Neutral face is approximately around 0.55 here.
        pitch = (
            48.0
            - nose_position * 48.0
        )

        yaw = float(
            np.clip(
                yaw,
                -60.0,
                60.0
            )
        )

        pitch = float(
            np.clip(
                pitch,
                -45.0,
                45.0
            )
        )

        roll = float(
            np.clip(
                roll,
                -45.0,
                45.0
            )
        )

        return np.asarray(
            [
                yaw,
                pitch,
                roll,
            ],
            dtype=np.float32
        )

    def _landmarks_to_array(
        self,
        landmarks
    ):
        try:

            output = []

            for point in landmarks:

                if isinstance(
                    point,
                    dict
                ):

                    x = point.get(
                        "x"
                    )

                    y = point.get(
                        "y"
                    )

                    z = point.get(
                        "z",
                        0.0
                    )

                else:

                    x = getattr(
                        point,
                        "x",
                        None
                    )

                    y = getattr(
                        point,
                        "y",
                        None
                    )

                    z = getattr(
                        point,
                        "z",
                        0.0
                    )

                    if x is None:

                        try:
                            x = point[0]
                            y = point[1]

                            if len(point) > 2:
                                z = point[2]

                        except Exception:
                            return None

                if x is None or y is None:
                    return None

                output.append(
                    [
                        float(x),
                        float(y),
                        float(z),
                    ]
                )

            array = np.asarray(
                output,
                dtype=np.float32
            )

            if array.ndim != 2:
                return None

            if array.shape[1] < 2:
                return None

            return array

        except Exception:

            return None

    # ==========================================================
    # POSE DISTANCE
    # ==========================================================

    def _pose_distance(
        self,
        stored_pose,
        live_pose
    ):
        stored_pose = np.asarray(
            stored_pose,
            dtype=np.float32
        )

        live_pose = np.asarray(
            live_pose,
            dtype=np.float32
        )

        # More tolerance for yaw because it generally changes
        # more strongly in the camera than pitch/roll.
        scale = np.asarray(
            [
                25.0,
                20.0,
                20.0,
            ],
            dtype=np.float32
        )

        difference = (
            stored_pose
            - live_pose
        )

        # Roll is angular and wraps around.
        difference[:, 2] = (
            (
                difference[:, 2]
                + 180.0
            )
            % 360.0
        ) - 180.0

        normalized = (
            difference / scale
        )

        distance = np.sqrt(
            np.sum(
                normalized
                * normalized,
                axis=1
            )
            / 3.0
        )

        return distance

    # ==========================================================
    # EMBEDDING HELPERS
    # ==========================================================

    def _decode_embedding(
        self,
        value
    ):
        if value is None:
            return None

        if isinstance(
            value,
            np.ndarray
        ):
            return self._normalize_embedding(
                value
            )

        if isinstance(
            value,
            (
                bytes,
                bytearray,
                memoryview
            )
        ):

            raw = bytes(value)

            if len(raw) == 512 * 4:

                vector = np.frombuffer(
                    raw,
                    dtype=np.float32
                )

                return self._normalize_embedding(
                    vector
                )

            if len(raw) == 512 * 8:

                vector = np.frombuffer(
                    raw,
                    dtype=np.float64
                ).astype(
                    np.float32
                )

                return self._normalize_embedding(
                    vector
                )

            try:

                decoded = json.loads(
                    raw.decode("utf-8")
                )

                return self._normalize_embedding(
                    decoded
                )

            except Exception:

                return None

        if isinstance(
            value,
            str
        ):

            try:

                decoded = json.loads(
                    value
                )

                return self._normalize_embedding(
                    decoded
                )

            except Exception:

                return None

        return self._normalize_embedding(
            value
        )

    def _normalize_embedding(
        self,
        embedding
    ):
        try:

            vector = np.asarray(
                embedding,
                dtype=np.float32
            ).reshape(-1)

        except Exception:

            return None

        if vector.size != 512:
            return None

        if not np.all(
            np.isfinite(vector)
        ):
            return None

        norm = float(
            np.linalg.norm(vector)
        )

        if norm <= 1e-8:
            return None

        return vector / norm

    def _normalize_matrix(
        self,
        matrix
    ):
        try:

            matrix = np.asarray(
                matrix,
                dtype=np.float32
            )

        except Exception:

            return np.empty(
                (0, 512),
                dtype=np.float32
            )

        if matrix.ndim != 2:
            return np.empty(
                (0, 512),
                dtype=np.float32
            )

        if matrix.shape[1] != 512:
            return np.empty(
                (0, 512),
                dtype=np.float32
            )

        finite_rows = np.all(
            np.isfinite(matrix),
            axis=1
        )

        matrix = matrix[
            finite_rows
        ]

        if matrix.shape[0] == 0:
            return np.empty(
                (0, 512),
                dtype=np.float32
            )

        norms = np.linalg.norm(
            matrix,
            axis=1,
            keepdims=True
        )

        valid = (
            norms[:, 0]
            > 1e-8
        )

        matrix = matrix[
            valid
        ]

        norms = norms[
            valid
        ]

        if matrix.shape[0] == 0:
            return np.empty(
                (0, 512),
                dtype=np.float32
            )

        return matrix / norms

    # ==========================================================
    # PATH HELPERS
    # ==========================================================

    def _resolve_db_path(self):

        candidates = []

        for attribute in (
            "db_path",
            "database_path",
            "db_file",
        ):

            value = getattr(
                self.database,
                attribute,
                None
            )

            if value is not None:

                candidates.append(
                    Path(value)
                )

        data_directory = getattr(
            self.database,
            "data_directory",
            None
        )

        if data_directory is not None:

            candidates.append(
                Path(data_directory)
                / "omi.db"
            )

        for path in candidates:

            try:

                if path.exists():
                    return path

            except Exception:
                pass

        if candidates:
            return candidates[0]

        return None

    def _safe_name(
        self,
        name
    ):
        safe = "".join(
            character
            if (
                character.isalnum()
                or character in "_-"
            )
            else "_"
            for character in str(name)
        )

        safe = safe.strip("_")

        return safe or "Person"

    # ==========================================================
    # UNKNOWN
    # ==========================================================

    def _unknown_result(
        self,
        face_key
    ):
        state = self.state.setdefault(
            face_key,
            {
                "candidate": None,
                "candidate_count": 0,
                "identity": "Unknown",
                "unknown_streak": 0,
                "similarity": None,
                "margin": None,
            }
        )

        state["unknown_streak"] += 1
        state["candidate"] = None
        state["candidate_count"] = 0

        if (
            state["unknown_streak"]
            >= self.max_unknown_streak
        ):

            state["identity"] = "Unknown"
            state["similarity"] = None
            state["margin"] = None

        return {
            "name": state["identity"],
            "similarity": state["similarity"],
            "margin": state["margin"],
            "status": "unknown",
            "top_similarity": state["similarity"],
            "topk_similarity": None,
            "support_count": 0,
            "sample_count": 0,
            "second_best": None,
            "second_similarity": None,
            "pose_used": False,
            "pose_distance": None,
            "pose_bonus": 0.0,
            "live_pose": None,
        }