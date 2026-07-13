"""Tests for the pocket-tts FastAPI server (pocket_tts.main.web_app).

Most tests here use the real model, loaded once for the module (mirrors how
`serve` actually runs -- one shared instance, many requests), matching this
repo's existing "real, unmocked" test style (see test_cli_generate.py,
test_quantization.py).

The concurrency test is a deliberate, documented exception: it monkeypatches
`tts_model` with a lightweight fake so it can assert serialization ordering
deterministically and fast, rather than inferring it from wall-clock timing
against the real (slow, variable) model.
"""

import time
import threading

import pytest
import torch
from fastapi.testclient import TestClient

from pocket_tts import voice_profiles
from pocket_tts.data.audio import audio_read


@pytest.fixture(scope="module")
def real_model():
    from pocket_tts import main as main_module
    from pocket_tts.models.tts_model import TTSModel

    model = TTSModel.load_model()
    main_module.tts_model = model
    yield model
    main_module.tts_model = None


@pytest.fixture(scope="module")
def client(real_model):
    from pocket_tts.main import web_app

    return TestClient(web_app)


@pytest.fixture
def chemist_profile(isolated_cache, real_model):
    """A real profile (built from a built-in catalog voice, no gated model
    needed) with one rule attached, for exercising /tts end to end."""
    state = real_model.get_state_for_audio_prompt("alba")
    voice_profiles.save_profile("chemist", state, source="alba")
    voice_profiles.add_rule("chemist", "NaCl", "sodium chloride")
    return "chemist"


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}


def test_profiles_and_history_empty_in_isolated_cache(client, isolated_cache):
    assert client.get("/profiles").json() == []
    assert client.get("/history").json() == []


def test_tts_with_profile_returns_valid_audio(client, chemist_profile, tmp_path):
    response = client.post(
        "/tts", data={"text": "Hello from the test suite.", "voice_profile": chemist_profile}
    )
    assert response.status_code == 200

    out_path = tmp_path / "server_test.wav"
    out_path.write_bytes(response.content)
    audio, sample_rate = audio_read(str(out_path))
    assert sample_rate == 24000
    assert audio.shape[0] == 1
    assert audio.shape[1] > 0


def test_tts_mutually_exclusive_params_rejected(client, chemist_profile):
    response = client.post(
        "/tts",
        data={"text": "Test.", "voice_profile": chemist_profile, "voice_url": "alba"},
    )
    assert response.status_code == 400


def test_tts_unknown_profile_returns_404(client, isolated_cache):
    response = client.post("/tts", data={"text": "Test.", "voice_profile": "does_not_exist"})
    assert response.status_code == 404


def test_tts_applies_profile_rules_visible_in_history(client, chemist_profile):
    response = client.post(
        "/tts", data={"text": "The formula is NaCl.", "voice_profile": chemist_profile}
    )
    assert response.status_code == 200

    rows = client.get(f"/history?profile={chemist_profile}").json()
    assert len(rows) == 1
    assert rows[0]["text"] == "The formula is sodium chloride."


class _FakeConfig:
    class mimi:
        sample_rate = 1000


class FakeTTSModel:
    """Minimal stand-in exposing only what /tts touches, so the concurrency
    test can assert serialization ordering without a slow, variable-timing
    real generation."""

    def __init__(self, event_log):
        self.event_log = event_log
        self.origin = None
        self.config = _FakeConfig()

    def _cached_get_state_for_audio_prompt(self, path):
        return {}

    def generate_audio_stream(self, model_state, text_to_generate):
        self.event_log.append("enter")
        time.sleep(0.2)
        self.event_log.append("exit")
        yield torch.zeros(1, 10)


def test_concurrent_requests_are_serialized(client, isolated_cache, real_model, monkeypatch):
    from pocket_tts import main as main_module

    assert not main_module._generation_lock.locked(), (
        "lock should not be held before this test starts"
    )

    voice_profiles.save_profile("fake_voice", {"m": {"k": torch.zeros(2, 2)}}, source="x")

    event_log: list[str] = []
    monkeypatch.setattr(main_module, "tts_model", FakeTTSModel(event_log))

    results = {}

    def make_request(key):
        response = client.post("/tts", data={"text": "hi", "voice_profile": "fake_voice"})
        results[key] = response.status_code

    t1 = threading.Thread(target=make_request, args=("a",))
    t2 = threading.Thread(target=make_request, args=("b",))
    t1.start()
    t2.start()
    t1.join()
    t2.join()

    assert results == {"a": 200, "b": 200}

    # A second "enter" must never appear before its predecessor's "exit" --
    # i.e. generation never overlaps.
    depth = 0
    for event in event_log:
        depth += 1 if event == "enter" else -1
        assert depth <= 1, f"overlapping generation detected: {event_log}"
    assert event_log == ["enter", "exit", "enter", "exit"]
