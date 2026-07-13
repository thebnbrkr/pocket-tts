# Voice Profiles

Voice cloning from a raw audio file works, but re-pointing at a `.wav`/`.mp3` file
every time is slow (it has to re-encode the audio) and inconvenient to share between
the CLI, the server, and the web UI. **Voice profiles** let you clone a voice once,
give it a name, and reuse that name everywhere afterward.

A profile is stored as a `.safetensors` voice state (the same format `export-voice`
produces) plus a small JSON metadata file (tags, notes, source, language), saved under
`~/.cache/pocket_tts/profiles/`.

## Prerequisites

- Cloning from **your own audio file** requires the gated voice-cloning model. Accept
  the terms at [huggingface.co/kyutai/pocket-tts](https://huggingface.co/kyutai/pocket-tts),
  then log in locally with `uv run hf auth login` (do this in your own terminal — never
  paste a token into a chat or share it with anyone).
- Cloning from one of the **built-in catalog voices** (see the list in the [main docs](../index.md))
  does *not* require this — those use pre-computed embeddings.

## Step 1: Create a profile

```bash
uv run pocket-tts create-profile /path/to/your.wav myvoice --tags personal
```

- `audio-path`: local path, `http(s)://` URL, or `hf://` URL (or a built-in voice name like `alba`).
- `name`: what you'll refer to this profile as everywhere else (letters, digits, `_`, `-` only).
- `--tags`: optional, comma-separated (e.g. `--tags medical,calm`).
- `--notes`: optional free-form text.
- `--overwrite`: replace an existing profile with the same name.
- `--language` / `--config`: same as `generate`/`export-voice`, selects which model encodes the audio.

Only the first 30 seconds of the audio file are used.

## Step 2: List your profiles

```bash
uv run pocket-tts list-profiles
```

```
myvoice              lang=-          tags=personal
```

## Step 3: Generate speech from a profile

The plain `generate` command reads any `.safetensors` path via `--voice`, so point it
straight at the saved profile file:

```bash
uv run pocket-tts generate \
  --voice ~/.cache/pocket_tts/profiles/myvoice.safetensors \
  --text "Hello, this is my cloned voice." \
  --output-path result.wav
```

## Step 4: Use a profile from the server or web UI

Profiles are resolved **by name** (not by path) through the server, which is the more
convenient way to reuse them day-to-day:

```bash
uv run pocket-tts serve
```

- **Web UI**: open `http://localhost:8000` — saved profiles appear in the "Saved voice
  profile" dropdown automatically.
- **HTTP API**:

```bash
# list profiles as JSON
curl http://localhost:8000/profiles

# generate speech using a saved profile
curl -X POST http://localhost:8000/tts \
  -F "text=Hello from my cloned voice." \
  -F "voice_profile=myvoice" \
  -o out.wav
```

`voice_profile`, `voice_url`, and `voice_wav` are mutually exclusive on `/tts` — pass
exactly one.

## Step 5: Use a profile from Python

```python
from pocket_tts import TTSModel
from pocket_tts import voice_profiles

model = TTSModel.load_model()

# Create and save a profile
state = model.get_state_for_audio_prompt("/path/to/your.wav", truncate=True)
voice_profiles.save_profile("myvoice", state, source="/path/to/your.wav", tags=["personal"])

# Later, reload it by name and generate
path = voice_profiles.get_profile_path("myvoice")
state = model.get_state_for_audio_prompt(str(path))
audio = model.generate_audio(state, "Hello from my cloned voice.")
```

`voice_profiles.list_profiles()` and `voice_profiles.delete_profile(name)` are also
available for managing saved profiles programmatically.
