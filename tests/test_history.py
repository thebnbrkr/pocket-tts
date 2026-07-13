"""Tests for pocket_tts.history. Uses fake audio-chunk generators instead of
real generated audio, since log_generation/list_history only ever touch plain
Python values and track_and_log only cares about chunk shapes and timing."""

import pytest
import torch

from pocket_tts import history


def test_log_and_list_history_roundtrip(isolated_cache):
    history.log_generation(
        profile_name="chemist",
        voice_source="chemist",
        text="Hello world.",
        duration_ms=100,
        audio_duration_ms=200,
        source="cli",
    )
    rows = history.list_history()
    assert len(rows) == 1
    assert rows[0]["text"] == "Hello world."
    assert rows[0]["profile_name"] == "chemist"
    assert rows[0]["source"] == "cli"


def test_list_history_most_recent_first(isolated_cache):
    history.log_generation(
        profile_name=None,
        voice_source="alba",
        text="first",
        duration_ms=1,
        audio_duration_ms=1,
        source="cli",
    )
    history.log_generation(
        profile_name=None,
        voice_source="alba",
        text="second",
        duration_ms=1,
        audio_duration_ms=1,
        source="cli",
    )
    rows = history.list_history()
    assert [r["text"] for r in rows] == ["second", "first"]


def test_list_history_filters_by_profile(isolated_cache):
    history.log_generation(
        profile_name="chemist",
        voice_source="chemist",
        text="a",
        duration_ms=1,
        audio_duration_ms=1,
        source="server",
    )
    history.log_generation(
        profile_name="other",
        voice_source="other",
        text="b",
        duration_ms=1,
        audio_duration_ms=1,
        source="server",
    )
    rows = history.list_history(profile_name="chemist")
    assert len(rows) == 1
    assert rows[0]["text"] == "a"


def test_list_history_respects_limit(isolated_cache):
    for i in range(5):
        history.log_generation(
            profile_name=None,
            voice_source="alba",
            text=f"row {i}",
            duration_ms=1,
            audio_duration_ms=1,
            source="cli",
        )
    assert len(history.list_history(limit=2)) == 2
    assert len(history.list_history(limit=100)) == 5


def test_clear_history(isolated_cache):
    history.log_generation(
        profile_name=None,
        voice_source="alba",
        text="a",
        duration_ms=1,
        audio_duration_ms=1,
        source="cli",
    )
    history.clear_history()
    assert history.list_history() == []


def fake_chunks(n_chunks, samples_per_chunk):
    for _ in range(n_chunks):
        yield torch.zeros(1, samples_per_chunk)


def test_track_and_log_passes_chunks_through_unchanged(isolated_cache):
    wrapped = history.track_and_log(
        fake_chunks(3, 10),
        profile_name="chemist",
        voice_source="chemist",
        text="test",
        source="server",
        sample_rate=1000,
    )
    chunks = list(wrapped)
    assert len(chunks) == 3
    for chunk in chunks:
        assert chunk.shape == (1, 10)


def test_track_and_log_computes_audio_duration_correctly(isolated_cache):
    # 3 chunks of 100 samples each, at 1000 Hz -> 300 samples / 1000 Hz = 300ms
    list(
        history.track_and_log(
            fake_chunks(3, 100),
            profile_name="chemist",
            voice_source="chemist",
            text="test",
            source="server",
            sample_rate=1000,
        )
    )
    rows = history.list_history()
    assert len(rows) == 1
    assert rows[0]["audio_duration_ms"] == 300
    assert rows[0]["duration_ms"] is not None


def test_track_and_log_still_logs_when_generator_raises(isolated_cache):
    def failing_chunks():
        yield torch.zeros(1, 10)
        raise RuntimeError("simulated generation failure")

    with pytest.raises(RuntimeError):
        list(
            history.track_and_log(
                failing_chunks(),
                profile_name="chemist",
                voice_source="chemist",
                text="test",
                source="server",
                sample_rate=1000,
            )
        )

    # despite the failure, a row should still have been logged (finally block)
    rows = history.list_history()
    assert len(rows) == 1
