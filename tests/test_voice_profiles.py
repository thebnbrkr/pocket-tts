"""Tests for pocket_tts.voice_profiles. Uses a fake model_state (a plain dict of
tensors) instead of loading the real TTSModel -- save_profile only needs
something export_model_state (pure safetensors I/O) can write, so these tests
run fast and need no network/model download."""

import json
import uuid

import pytest
import torch

from pocket_tts import voice_profiles


def fake_model_state():
    return {"module": {"key": torch.zeros(2, 2)}}


def test_save_profile_writes_expected_fields(isolated_cache):
    voice_profiles.save_profile("myvoice", fake_model_state(), source="test.wav", tags=["a"])

    profiles = voice_profiles.list_profiles()
    assert len(profiles) == 1
    p = profiles[0]
    assert p["name"] == "myvoice"
    assert p["source"] == "test.wav"
    assert p["tags"] == ["a"]
    assert p["rules"] == []
    assert uuid.UUID(p["id"])  # valid uuid4 string

    safetensors_path = voice_profiles.get_profile_path("myvoice")
    assert safetensors_path.exists()


def test_get_profile_path_unknown_raises(isolated_cache):
    with pytest.raises(voice_profiles.ProfileNotFoundError):
        voice_profiles.get_profile_path("nope")


def test_invalid_name_rejected(isolated_cache):
    with pytest.raises(ValueError):
        voice_profiles.save_profile("bad name!", fake_model_state(), source="x")


def test_save_profile_requires_overwrite_flag(isolated_cache):
    voice_profiles.save_profile("dup", fake_model_state(), source="x")
    with pytest.raises(FileExistsError):
        voice_profiles.save_profile("dup", fake_model_state(), source="y")

    # overwrite=True succeeds and replaces metadata
    voice_profiles.save_profile("dup", fake_model_state(), source="y", overwrite=True)
    assert voice_profiles.list_profiles()[0]["source"] == "y"


def test_add_list_remove_rule_roundtrip(isolated_cache):
    voice_profiles.save_profile("chemist", fake_model_state(), source="x")
    voice_profiles.add_rule("chemist", "NaCl", "sodium chloride")

    rules = voice_profiles.list_rules("chemist")
    assert rules == [{"pattern": "NaCl", "replacement": "sodium chloride", "regex": False}]

    voice_profiles.remove_rule("chemist", 0)
    assert voice_profiles.list_rules("chemist") == []


def test_remove_rule_bad_index_raises(isolated_cache):
    voice_profiles.save_profile("chemist", fake_model_state(), source="x")
    with pytest.raises(IndexError):
        voice_profiles.remove_rule("chemist", 0)


def test_add_rules_bulk(isolated_cache):
    voice_profiles.save_profile("chemist", fake_model_state(), source="x")
    count = voice_profiles.add_rules(
        "chemist",
        [
            {"pattern": "NaCl", "replacement": "sodium chloride"},
            {"pattern": "H2O", "replacement": "water"},
        ],
    )
    assert count == 2
    assert len(voice_profiles.list_rules("chemist")) == 2


def test_add_rules_bulk_atomic_on_bad_entry(isolated_cache):
    """A malformed entry must fail the whole import, not partially apply it."""
    voice_profiles.save_profile("chemist", fake_model_state(), source="x")
    with pytest.raises(ValueError, match="index 1"):
        voice_profiles.add_rules(
            "chemist",
            [
                {"pattern": "NaCl", "replacement": "sodium chloride"},
                {"pattern": "missing_replacement"},
            ],
        )
    assert voice_profiles.list_rules("chemist") == []


def test_remove_rules_bulk_handles_index_shift(isolated_cache):
    """Removing indices [0, 2] from a 3-item list must leave exactly the
    middle item -- regression test for the classic pop-while-iterating bug."""
    voice_profiles.save_profile("chemist", fake_model_state(), source="x")
    voice_profiles.add_rules(
        "chemist",
        [
            {"pattern": "a", "replacement": "A"},
            {"pattern": "b", "replacement": "B"},
            {"pattern": "c", "replacement": "C"},
        ],
    )
    voice_profiles.remove_rules("chemist", [0, 2])
    remaining = voice_profiles.list_rules("chemist")
    assert len(remaining) == 1
    assert remaining[0]["pattern"] == "b"


def test_clear_rules(isolated_cache):
    voice_profiles.save_profile("chemist", fake_model_state(), source="x")
    voice_profiles.add_rule("chemist", "NaCl", "sodium chloride")
    voice_profiles.clear_rules("chemist")
    assert voice_profiles.list_rules("chemist") == []


def test_apply_rules_literal_whole_word_case_insensitive(isolated_cache):
    voice_profiles.save_profile("chemist", fake_model_state(), source="x")
    voice_profiles.add_rule("chemist", "NaCl", "sodium chloride")

    assert (
        voice_profiles.apply_rules("chemist", "The formula is nacl.")
        == "The formula is sodium chloride."
    )
    # whole-word: shouldn't match inside a longer token
    assert voice_profiles.apply_rules("chemist", "NaClX is different") == "NaClX is different"


def test_apply_rules_regex_mode(isolated_cache):
    voice_profiles.save_profile("chemist", fake_model_state(), source="x")
    voice_profiles.add_rule("chemist", r"(\d+)mg", r"\1 milligrams", regex=True)

    assert voice_profiles.apply_rules("chemist", "Take 50mg daily.") == "Take 50 milligrams daily."


def test_apply_rules_no_rules_returns_text_unchanged(isolated_cache):
    voice_profiles.save_profile("empty", fake_model_state(), source="x")
    assert voice_profiles.apply_rules("empty", "Hello world.") == "Hello world."


def test_apply_rules_isolated_between_profiles(isolated_cache):
    voice_profiles.save_profile("chemist", fake_model_state(), source="x")
    voice_profiles.save_profile("other", fake_model_state(), source="x")
    voice_profiles.add_rule("chemist", "NaCl", "sodium chloride")

    assert voice_profiles.apply_rules("other", "The formula is NaCl.") == "The formula is NaCl."


def test_backward_compat_profile_without_id_or_rules_fields(isolated_cache):
    """Simulate a profile saved before the id/rules fields existed."""
    voice_profiles.save_profile("legacy", fake_model_state(), source="x")
    metadata_path = isolated_cache / "profiles" / "legacy.json"
    old_style = {"name": "legacy", "source": "x", "language": None, "tags": [], "notes": ""}
    metadata_path.write_text(json.dumps(old_style))

    assert voice_profiles.list_rules("legacy") == []
    assert voice_profiles.apply_rules("legacy", "unchanged text") == "unchanged text"
    profiles = voice_profiles.list_profiles()
    assert any(p["name"] == "legacy" for p in profiles)


def test_delete_profile_removes_files_and_is_idempotent(isolated_cache):
    voice_profiles.save_profile("gone", fake_model_state(), source="x")
    voice_profiles.delete_profile("gone")
    assert voice_profiles.list_profiles() == []

    # calling again on an already-deleted profile must not raise
    voice_profiles.delete_profile("gone")
