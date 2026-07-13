import os

import pytest

os.environ["POCKET_TTS_ERROR_WITHOUT_EOS"] = "1"


@pytest.fixture(autouse=True)
def isolated_cache(tmp_path, monkeypatch):
    """Redirect voice_profiles/history storage to a temp dir so no test --
    including existing ones that invoke the `generate` CLI command, which
    logs to history as a side effect -- ever writes to the real
    ~/.cache/pocket_tts/ directory. autouse=True so this applies to every
    test in the suite, not just ones that request it by name.

    Both modules do `from ...utils import make_cache_directory`, which binds a
    separate name in each module's own namespace -- patching
    `pocket_tts.utils.utils.make_cache_directory` would NOT affect either of
    them, so each is patched at its own point of use. This does NOT affect
    the separate model/tokenizer download cache in utils.make_cache_directory
    itself, so existing tests still reuse already-downloaded model weights.
    """
    from pocket_tts import history, voice_profiles

    monkeypatch.setattr(voice_profiles, "make_cache_directory", lambda: tmp_path)
    monkeypatch.setattr(history, "HISTORY_DB_PATH", tmp_path / "history.db")
    return tmp_path
