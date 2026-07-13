# Serve

The `serve` command starts a FastAPI web server that provides both a web interface and HTTP API for text-to-speech generation.

## Basic Usage

```bash
uvx pocket-tts serve
# or if installed manually:
pocket-tts serve
```

This starts a server on `http://localhost:8000` with the default voice model.

## Command Options

- `--host HOST`: Host to bind to (default: "localhost")
- `--port PORT`: Port to bind to (default: 8000)
- `--reload`: Enable auto-reload for development
- `--language`: Language for the TTS model, one of `'english_2026-01'`, `'english_2026-04'`, `'english'`, `'french_24l'`, `'german_24l'`, `'portuguese_24l'`, `'italian_24l'`, `'spanish_24l'` (default: `english`, which is the same model as `'english_2026-04'`). Incompatible with `--config`. The "24l" variants are bigger models, not distilled yet and here only as preview.
- `--config`: Path to a custom config .yaml. Incompatible with `--language`.
- `--quantize`: Use int8 quantization for the model (default: False). This can reduce memory usage and increase speed, with minimal impact on audio quality.
## Examples

### Basic Server

```bash
# Start with default settings
pocket-tts serve

# Custom host and port
pocket-tts serve --host "localhost" --port 8080
```

### Custom Language
To select the default language model, pass `--language`:
```bash
pocket-tts serve --language french_24l
```

### Custom Model Config

If you'd like to override the paths from which the models are loaded, you can provide a custom YAML configuration.

Copy one of the files in `pocket_tts/config` (for example `pocket_tts/config/english.yaml`) and change `weights_path`, `weights_path_without_voice_cloning:`, and `tokenizer_path:` to the paths of the models you want to load.

Then, use the --config option to point to your newly created config.

```bash
# Use a different config
pocket-tts serve --config "C://pocket-tts/my_config.yaml"
```

## Web Interface

Once the server is running, navigate to `http://localhost:8000` to access the web interface.

### Diagnostics: the History panel

Below the generate form, the web UI shows a live table of everything generated so far
(fed entirely by the existing [`/history`](history.md) endpoint — no separate
tracking, it's the same data the CLI's `pocket-tts history` shows). Each row shows
when it ran, whether it came from the CLI or the server, which voice/profile was
used, a snippet of the text (hover for the full text), and the audio duration.
It refreshes automatically after every generation, and has a manual "Refresh"
button. If [pronunciation checking](pronunciation_check.md) is enabled, rows where
the generated audio didn't match the requested text get a red "⚠ mismatch" badge,
with the actual transcript shown on hover — this can appear a moment *after* the
row first shows up, since that check runs in the background (see below), not
before the response is sent.

## Concurrency

`TTSModel` is not thread-safe (see the [Python API docs](python-api.md)), and the
server holds a single shared model instance across all requests. Concurrent `/tts`
requests are safely serialized behind an internal lock rather than running in
parallel — if two requests arrive at the same time, the second one's generation
simply waits for the first to finish before starting, rather than corrupting output.
Other endpoints (`/health`, `/profiles`, `/history`) are unaffected and stay
responsive even while a generation is in progress.

For more advanced usage, see the [Python API documentation](python-api.md) for direct integration with the TTS model.
