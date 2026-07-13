import sqlite3
import time
from datetime import datetime, timezone

from pocket_tts.utils.utils import make_cache_directory

HISTORY_DB_PATH = make_cache_directory() / "history.db"


def _ensure_schema(conn: sqlite3.Connection) -> None:
    """Add columns introduced after the table already existed on disk.

    CREATE TABLE IF NOT EXISTS is a no-op against a database that already has
    the `generations` table (e.g. this repo's own real history.db) -- new
    columns must be migrated in explicitly, or they'd silently never appear
    on any pre-existing database.
    """
    existing_columns = {row[1] for row in conn.execute("PRAGMA table_info(generations)")}
    for column, column_type in [("transcribed_text", "TEXT"), ("pronunciation_match", "INTEGER")]:
        if column not in existing_columns:
            conn.execute(f"ALTER TABLE generations ADD COLUMN {column} {column_type}")


def _get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(HISTORY_DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS generations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT NOT NULL,
            profile_name TEXT,
            voice_source TEXT,
            text TEXT NOT NULL,
            duration_ms INTEGER,
            audio_duration_ms INTEGER,
            source TEXT NOT NULL
        )
        """
    )
    _ensure_schema(conn)
    return conn


def log_generation(
    *,
    profile_name: str | None,
    voice_source: str | None,
    text: str,
    duration_ms: int | None,
    audio_duration_ms: int | None,
    source: str,
) -> int:
    with _get_connection() as conn:
        cursor = conn.execute(
            """
            INSERT INTO generations
                (created_at, profile_name, voice_source, text, duration_ms, audio_duration_ms, source)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                datetime.now(timezone.utc).isoformat(),
                profile_name,
                voice_source,
                text,
                duration_ms,
                audio_duration_ms,
                source,
            ),
        )
        return cursor.lastrowid


def list_history(profile_name: str | None = None, limit: int = 50) -> list[dict]:
    with _get_connection() as conn:
        if profile_name is not None:
            rows = conn.execute(
                "SELECT * FROM generations WHERE profile_name = ? ORDER BY id DESC LIMIT ?",
                (profile_name, limit),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM generations ORDER BY id DESC LIMIT ?", (limit,)
            ).fetchall()
        return [dict(row) for row in rows]


def clear_history() -> None:
    with _get_connection() as conn:
        conn.execute("DELETE FROM generations")


def update_transcription(row_id: int, transcribed_text: str, matched: bool) -> None:
    with _get_connection() as conn:
        conn.execute(
            "UPDATE generations SET transcribed_text = ?, pronunciation_match = ? WHERE id = ?",
            (transcribed_text, int(matched), row_id),
        )


def track_and_log(
    audio_chunks,
    *,
    profile_name: str | None,
    voice_source: str | None,
    text: str,
    source: str,
    sample_rate: int,
    row_id_out: list | None = None,
):
    """Passthrough generator: yields chunks unchanged, logs once the stream ends.

    Generators consumed via a plain for-loop (as stream_audio_chunks does)
    can't return a value the normal way, so the logged row's id is instead
    written into `row_id_out` (if given) once the finally block runs -- the
    caller reads it after fully draining this generator.
    """
    total_samples = 0
    start_time = time.monotonic()
    try:
        for chunk in audio_chunks:
            total_samples += chunk.shape[-1]
            yield chunk
    finally:
        duration_ms = int((time.monotonic() - start_time) * 1000)
        audio_duration_ms = int(total_samples * 1000 / sample_rate)
        row_id = log_generation(
            profile_name=profile_name,
            voice_source=voice_source,
            text=text,
            duration_ms=duration_ms,
            audio_duration_ms=audio_duration_ms,
            source=source,
        )
        if row_id_out is not None:
            row_id_out.append(row_id)
