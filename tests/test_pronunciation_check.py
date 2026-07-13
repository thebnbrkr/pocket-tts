"""Tests for pocket_tts.pronunciation_check. Uses a fake ASR model instead of
loading real faster-whisper, so these tests are fast and need no extra
dependency installed -- mirrors how the concurrency test in test_server.py
used a fake tts_model instead of a slow real one."""

import sqlite3

import torch

from pocket_tts import history, pronunciation_check


class FakeSegment:
    def __init__(self, text):
        self.text = text


class FakeASRModel:
    def __init__(self, transcript):
        self.transcript = transcript

    def transcribe(self, path):
        return [FakeSegment(self.transcript)], None


def test_normalize_strips_case_punctuation_whitespace():
    assert pronunciation_check._normalize("The Formula is NaCl!!") == "the formula is nacl"
    assert pronunciation_check._normalize("  extra   spaces ") == "extra spaces"


def test_check_pronunciation_records_match(isolated_cache, monkeypatch):
    row_id = history.log_generation(
        profile_name="chemist",
        voice_source="chemist",
        text="sodium chloride",
        duration_ms=1,
        audio_duration_ms=1,
        source="server",
    )
    monkeypatch.setattr(
        pronunciation_check, "_get_asr_model", lambda: FakeASRModel("Sodium chloride.")
    )

    pronunciation_check.check_pronunciation(torch.zeros(1000), 16000, "sodium chloride", row_id)

    row = history.list_history()[0]
    assert row["transcribed_text"] == "Sodium chloride."
    assert row["pronunciation_match"] == 1


def test_check_pronunciation_records_mismatch(isolated_cache, monkeypatch):
    row_id = history.log_generation(
        profile_name="chemist",
        voice_source="chemist",
        text="sodium chloride",
        duration_ms=1,
        audio_duration_ms=1,
        source="server",
    )
    monkeypatch.setattr(
        pronunciation_check, "_get_asr_model", lambda: FakeASRModel("sodium cyanide")
    )

    pronunciation_check.check_pronunciation(torch.zeros(1000), 16000, "sodium chloride", row_id)

    row = history.list_history()[0]
    assert row["transcribed_text"] == "sodium cyanide"
    assert row["pronunciation_match"] == 0


def test_check_pronunciation_never_raises_on_failure(isolated_cache, monkeypatch):
    def broken_model():
        raise RuntimeError("model failed to load")

    monkeypatch.setattr(pronunciation_check, "_get_asr_model", broken_model)

    # Must not raise -- this runs in a fire-and-forget background thread with
    # nothing waiting on the result, so failures can only be logged.
    pronunciation_check.check_pronunciation(torch.zeros(1000), 16000, "text", row_id=999)


def test_migration_adds_columns_to_pre_existing_database(isolated_cache):
    """Simulates this repo's own real history.db (created before these
    columns existed) to confirm the migration is safe against it."""
    db_path = isolated_cache / "history.db"
    conn = sqlite3.connect(db_path)
    conn.execute(
        """
        CREATE TABLE generations (
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
    conn.execute(
        "INSERT INTO generations (created_at, profile_name, voice_source, text, "
        "duration_ms, audio_duration_ms, source) VALUES "
        "('t', 'p', 'v', 'pre-existing row', 1, 1, 'cli')"
    )
    conn.commit()
    conn.close()

    rows = history.list_history()
    assert len(rows) == 1
    assert rows[0]["text"] == "pre-existing row"
    assert rows[0]["transcribed_text"] is None
    assert rows[0]["pronunciation_match"] is None

    history.update_transcription(rows[0]["id"], "heard this", True)
    updated = history.list_history()[0]
    assert updated["transcribed_text"] == "heard this"
    assert updated["pronunciation_match"] == 1
