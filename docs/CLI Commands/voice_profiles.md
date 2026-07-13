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

## Step 6: Programmable rules (per-profile vocabulary/pronunciation overrides)

Every profile can carry its own list of text-substitution rules, applied automatically
to the text *before* generation whenever that profile is used via `/tts`. This is how
you give a profile domain vocabulary — e.g. a "chemist" profile that always expands
`NaCl` to `sodium chloride` — without affecting any other profile.

Rules live inside that profile's own metadata file (`~/.cache/pocket_tts/profiles/<name>.json`)
— there's no shared or global rule table, so a rule added to one profile has zero effect
on any other.

### Add a rule

```bash
uv run pocket-tts add-rule chemist "NaCl" "sodium chloride"
```

By default this is a whole-word, case-insensitive literal match (won't accidentally
match inside a longer word). For pattern-based matching, add `--regex`:

```bash
uv run pocket-tts add-rule chemist "(\d+)mg" "\1 milligrams" --regex
```

`add-rule` is additive — call it again for each new term you want covered; earlier
rules aren't touched.

### List rules for a profile

```bash
uv run pocket-tts list-rules chemist
```

```
[0] 'NaCl' -> 'sodium chloride'  (regex=False)
[1] '(\\d+)mg' -> '\\1 milligrams'  (regex=True)
```

### Remove a rule

```bash
uv run pocket-tts remove-rule chemist 0
```

Removes by index (shown in `list-rules`). There's no bulk-add/import command yet —
each rule is added one at a time via `add-rule`.

### How matching works

Rules are applied in the order they were added, each one scanning the *entire* text
being generated (not a word-by-word dictionary lookup) — so rule 2 sees whatever rule 1
already produced. Fine for the handful-to-dozens-of-rules scale this is meant for.

### From Python

```python
from pocket_tts import voice_profiles

voice_profiles.add_rule("chemist", "NaCl", "sodium chloride")
voice_profiles.list_rules("chemist")
voice_profiles.remove_rule("chemist", 0)

# apply_rules() is what /tts calls internally before generation
text = voice_profiles.apply_rules("chemist", "The formula is NaCl.")
# -> "The formula is sodium chloride."
```

### Note: server-only for now

Rule application is currently wired into the server's `voice_profile=` resolution path
only. The CLI's `generate --voice <path>` doesn't do profile-name lookup (see Step 3
above — it takes a raw `.safetensors` path), so it doesn't apply rules either.

### Profile IDs

Every profile also gets a stable `id` (UUID) at creation time, visible in its metadata
JSON. Profiles are still looked up by `name` everywhere today — the `id` field is there
for forward compatibility (e.g. a future tool that needs a stable reference even if a
profile gets renamed later), not yet a usable lookup key on its own.
