# Generation History

Every time you generate speech — from the CLI or the server — pocket-tts logs it
to a local SQLite database: when it happened, which voice/profile was used, the
full text that was spoken, and how long it took. This is useful for debugging,
for seeing what's actually been generated in a session, and as groundwork for
future tooling (e.g. a diagnostics view) built on top of the same data.

No setup required — logging happens automatically. The database lives at
`~/.cache/pocket_tts/history.db` (plain SQLite; open it with the `sqlite3` CLI
or any SQLite browser if you want to query it directly).

## What gets logged

Each generation is one row:

| Field | Description |
|---|---|
| `created_at` | UTC timestamp (ISO 8601) |
| `profile_name` | Set only if a saved [voice profile](voice_profiles.md) was used, otherwise `null` |
| `voice_source` | The raw voice value used — profile name, voice URL, built-in voice name, or `"upload"` |
| `text` | The full text that was generated, verbatim |
| `duration_ms` | Wall-clock generation time |
| `audio_duration_ms` | Length of the generated audio |
| `source` | `"cli"` or `"server"` |

Note: the full generated text is stored, not just metadata. If you're generating
anything sensitive, keep in mind it's persisted to disk in plain SQLite with no
redaction — there's no PII scrubbing layer yet.

## Viewing history from the CLI

```bash
pocket-tts history
```

```
2026-07-13T11:58:22.506462+00:00  server  voice=demo_voice            "History test two."
2026-07-13T11:57:29.432291+00:00  cli     voice=alba                  "History test one."
```

Options:

- `--profile NAME`: only show generations that used a specific saved profile.
- `--limit N`: max rows to show (default 20).

```bash
pocket-tts history --profile timdillon --limit 5
```

## Viewing history from the server

```bash
uv run pocket-tts serve
```

```bash
# all recent history, as JSON
curl http://localhost:8000/history

# filtered by profile, with a custom limit
curl "http://localhost:8000/history?profile=timdillon&limit=10"
```

## Querying the raw database

Since it's just SQLite, you're not limited to the CLI/API views:

```bash
sqlite3 ~/.cache/pocket_tts/history.db \
  "SELECT created_at, profile_name, duration_ms, audio_duration_ms FROM generations ORDER BY id DESC LIMIT 10;"
```

## Python API

```python
from pocket_tts import history

# most recent 50 generations, optionally filtered by profile
rows = history.list_history(profile_name="timdillon", limit=50)

# wipe all history
history.clear_history()
```
