import json
import re
import shutil
import sqlite3
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np


class FaceEmbeddingDatabase:

    def __init__(self):

        project_root = (
            Path(__file__).resolve().parent.parent
        )

        self.data_directory = (
            project_root / "data"
        )

        self.people_directory = (
            self.data_directory / "people"
        )

        self.database_path = (
            self.data_directory / "omi.db"
        )

        self.data_directory.mkdir(
            parents=True,
            exist_ok=True
        )

        self.people_directory.mkdir(
            parents=True,
            exist_ok=True
        )

        self.connection = sqlite3.connect(
            str(self.database_path)
        )

        self.connection.row_factory = (
            sqlite3.Row
        )

        self._create_tables()

    # =====================================================
    # TABLES
    # =====================================================

    def _create_tables(self):

        cursor = self.connection.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS people (

                id INTEGER PRIMARY KEY AUTOINCREMENT,

                name TEXT NOT NULL UNIQUE,

                created_at TEXT NOT NULL,

                profile_image TEXT,

                age_estimate REAL,

                age_confidence REAL,

                attributes_json TEXT
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS face_samples (

                id INTEGER PRIMARY KEY AUTOINCREMENT,

                person_id INTEGER NOT NULL,

                image_path TEXT NOT NULL,

                embedding BLOB NOT NULL,

                created_at TEXT NOT NULL,

                FOREIGN KEY(person_id)
                    REFERENCES people(id)
                    ON DELETE CASCADE
            )
        """)

        self.connection.commit()

    # =====================================================
    # SAFE DIRECTORY NAME
    # =====================================================

    def _safe_name(
        self,
        name
    ):

        safe = re.sub(
            r"[^a-zA-Z0-9_-]+",
            "_",
            name.strip()
        )

        safe = safe.strip("_")

        if not safe:

            safe = "person"

        return safe

    # =====================================================
    # EMBEDDING → BLOB
    # =====================================================

    def _embedding_to_blob(
        self,
        embedding
    ):

        array = np.asarray(
            embedding,
            dtype=np.float32
        )

        return sqlite3.Binary(
            array.tobytes()
        )

    # =====================================================
    # BLOB → EMBEDDING
    # =====================================================

    def _blob_to_embedding(
        self,
        blob
    ):

        array = np.frombuffer(
            blob,
            dtype=np.float32
        )

        return array.tolist()

    # =====================================================
    # ADD PERSON
    # =====================================================

    def add_person(
        self,
        name,
        samples
    ):

        name = name.strip()

        if not name:
            return False

        if not samples:
            return False

        cursor = self.connection.cursor()

        # -------------------------------------------------
        # Replace existing person
        # -------------------------------------------------

        cursor.execute(
            """
            SELECT id
            FROM people
            WHERE LOWER(name) = LOWER(?)
            """,
            (name,)
        )

        existing = cursor.fetchone()

        if existing:

            old_person_id = existing["id"]

            old_directory = (
                self.people_directory
                / (
                    f"person_{old_person_id:04d}_"
                    f"{self._safe_name(name)}"
                )
            )

            cursor.execute(
                """
                DELETE FROM face_samples
                WHERE person_id = ?
                """,
                (old_person_id,)
            )

            cursor.execute(
                """
                DELETE FROM people
                WHERE id = ?
                """,
                (old_person_id,)
            )

            self.connection.commit()

            if old_directory.exists():

                shutil.rmtree(
                    old_directory,
                    ignore_errors=True
                )

        # -------------------------------------------------
        # Create person
        # -------------------------------------------------

        created_at = datetime.now().isoformat(
            timespec="seconds"
        )

        cursor.execute(
            """
            INSERT INTO people (
                name,
                created_at,
                profile_image
            )
            VALUES (?, ?, ?)
            """,
            (
                name,
                created_at,
                None
            )
        )

        person_id = cursor.lastrowid

        person_directory = (
            self.people_directory
            / (
                f"person_{person_id:04d}_"
                f"{self._safe_name(name)}"
            )
        )

        samples_directory = (
            person_directory / "samples"
        )

        samples_directory.mkdir(
            parents=True,
            exist_ok=True
        )

        first_image_path = None
        saved_count = 0

        try:

            for index, sample in enumerate(
                samples,
                start=1
            ):

                embedding = sample.get(
                    "embedding"
                )

                image = sample.get(
                    "image"
                )

                if (
                    embedding is None
                    or image is None
                ):

                    continue

                if image.size == 0:

                    continue

                filename = (
                    f"sample_{index:03d}.jpg"
                )

                image_path = (
                    samples_directory
                    / filename
                )

                success = cv2.imwrite(
                    str(image_path),
                    image,
                    [
                        cv2.IMWRITE_JPEG_QUALITY,
                        95
                    ]
                )

                if not success:

                    continue

                relative_path = (
                    image_path.relative_to(
                        self.data_directory
                    )
                )

                if first_image_path is None:

                    first_image_path = (
                        relative_path
                    )

                cursor.execute(
                    """
                    INSERT INTO face_samples (
                        person_id,
                        image_path,
                        embedding,
                        created_at
                    )
                    VALUES (?, ?, ?, ?)
                    """,
                    (
                        person_id,
                        str(relative_path),
                        self._embedding_to_blob(
                            embedding
                        ),
                        created_at
                    )
                )

                saved_count += 1

            if (
                first_image_path is None
                or saved_count == 0
            ):

                raise RuntimeError(
                    "No valid face samples "
                    "were saved."
                )

            # Temporary image reference.
            cursor.execute(
                """
                UPDATE people
                SET profile_image = ?
                WHERE id = ?
                """,
                (
                    str(first_image_path),
                    person_id
                )
            )

            self.connection.commit()

        except Exception:

            self.connection.rollback()

            if person_directory.exists():

                shutil.rmtree(
                    person_directory,
                    ignore_errors=True
                )

            raise

        return True

    # =====================================================
    # UPDATE PROFILE CARD
    # =====================================================

    def set_profile_image(
        self,
        name,
        profile_path
    ):

        cursor = self.connection.cursor()

        cursor.execute(
            """
            UPDATE people
            SET profile_image = ?
            WHERE LOWER(name) = LOWER(?)
            """,
            (
                str(profile_path),
                name
            )
        )

        self.connection.commit()

        return cursor.rowcount > 0

    # =====================================================
    # GET PERSON
    # =====================================================

    def get_person(
        self,
        name
    ):

        cursor = self.connection.cursor()

        cursor.execute(
            """
            SELECT *
            FROM people
            WHERE LOWER(name) = LOWER(?)
            """,
            (name,)
        )

        row = cursor.fetchone()

        if row is None:

            return None

        return dict(row)

    # =====================================================
    # GET ALL PEOPLE
    # =====================================================

    def get_people(self):

        cursor = self.connection.cursor()

        cursor.execute(
            """
            SELECT
                id,
                name,
                created_at,
                profile_image,
                age_estimate,
                age_confidence,
                attributes_json
            FROM people
            ORDER BY id
            """
        )

        rows = cursor.fetchall()

        people = []

        for row in rows:

            person_id = row["id"]

            cursor.execute(
                """
                SELECT embedding
                FROM face_samples
                WHERE person_id = ?
                ORDER BY id
                """,
                (person_id,)
            )

            sample_rows = cursor.fetchall()

            embeddings = []

            for sample in sample_rows:

                embeddings.append(
                    self._blob_to_embedding(
                        sample["embedding"]
                    )
                )

            attributes = {}

            if row["attributes_json"]:

                try:

                    attributes = json.loads(
                        row["attributes_json"]
                    )

                except json.JSONDecodeError:

                    attributes = {}

            people.append({

                "id": person_id,

                "name": row["name"],

                "created_at": row[
                    "created_at"
                ],

                "profile_image": row[
                    "profile_image"
                ],

                "age_estimate": row[
                    "age_estimate"
                ],

                "age_confidence": row[
                    "age_confidence"
                ],

                "attributes": attributes,

                "embeddings": embeddings
            })

        return people

    # =====================================================
    # COUNT
    # =====================================================

    def count(self):

        cursor = self.connection.cursor()

        cursor.execute(
            "SELECT COUNT(*) FROM people"
        )

        return cursor.fetchone()[0]

    # =====================================================
    # CLOSE
    # =====================================================

    def close(self):

        if self.connection:

            self.connection.close()

            self.connection = None