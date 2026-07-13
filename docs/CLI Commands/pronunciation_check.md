# Pronunciation Check (ASR round-trip QA)

Pocket TTS doesn't know whether it actually said what you asked it to say —
it just generates audio. This feature closes that loop: after generating
speech, it transcribes the audio back to text using a *separate* speech-to-text
model, compares that transcript to the text you originally asked for, and
flags the result in [`/history`](history.md) if they don't match. This is how
you catch mispronunciations (slang, chemical formulas, unusual names, etc.)
automatically, without listening to every clip yourself.

## Why this is opt-in and off by default

This is a genuinely different kind of feature from everything else in the
server: it requires loading a **second model** (pocket-tts itself only does
text-to-speech; verifying what was said requires speech-to-text, a completely
separate model). That has real memory and startup cost, so it's not forced on
anyone by default.

```bash
pip install 'pocket-tts[asr]'
```

```bash
pocket-tts serve --enable-pronunciation-check
```

If you pass the flag without installing the extra, the server fails immediately
at startup with a clear error, rather than silently failing on the first request.

## How it works, mechanically

1. `/tts` generates and streams audio back to the client **exactly as before** —
   this feature adds zero latency to that response. The check only starts
   *after* the full response has already been sent.
2. In a background thread, the full generated audio is written to a temporary
   WAV file and transcribed using [faster-whisper](https://github.com/SYSTRAN/faster-whisper)
   (the `tiny` multilingual checkpoint, run on CPU with int8 quantization —
   chosen to match pocket-tts's own small, CPU-first design, unlike the full
   `openai-whisper` package which is much heavier to run on CPU).
3. Both the transcript and the original requested text are normalized (lowercased,
   punctuation stripped, whitespace collapsed) and compared for an exact match.
4. The result is written back into that generation's row in `history.db` via
   two new columns: `transcribed_text` (what was actually heard) and
   `pronunciation_match` (whether it matched).
5. The web UI's History panel (see [serve.md](serve.md)) shows a warning badge
   on any row where `pronunciation_match` is false, with the transcript visible
   on hover.

Because step 2 runs after the response is already gone, there's necessarily a
short window where a freshly generated row hasn't been checked yet — the flag
(or lack of one) appears a moment later once the background check finishes.

## A note on the database migration

If you already have a `history.db` from before this feature existed, the two
new columns are added automatically the next time anything touches history —
existing rows are preserved with `NULL` in the new columns rather than being
lost or requiring a manual migration step.

## From Python

```python
from pocket_tts import pronunciation_check

# normalization used for comparison
pronunciation_check._normalize("The Formula is NaCl!!")  # "the formula is nacl"
```

`check_pronunciation(audio, sample_rate, intended_text, row_id)` is the function
the server calls in its background thread; it's designed to never raise (failures
are logged, not propagated), since nothing is waiting on its result directly.

## Scope note

This is server-only for now — the plain `pocket-tts generate` CLI command does
not run a pronunciation check, matching how [per-profile rules](voice_profiles.md)
were scoped the same way initially.
