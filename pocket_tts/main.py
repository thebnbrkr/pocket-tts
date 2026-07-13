import io
import json
import logging
import os
import sys
import tempfile
import threading
from pathlib import Path
from queue import Queue

import torch
import typer
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse
from typing_extensions import Annotated

from pocket_tts.data.audio import stream_audio_chunks
from pocket_tts.default_parameters import (
    DEFAULT_EOS_THRESHOLD,
    DEFAULT_FRAMES_AFTER_EOS,
    DEFAULT_LSD_DECODE_STEPS,
    DEFAULT_NOISE_CLAMP,
    DEFAULT_TEMPERATURE,
    MAX_TOKEN_PER_CHUNK,
    get_default_text_for_language,
    get_default_voice_for_language,
)
from pocket_tts import history as history_module
from pocket_tts import pronunciation_check
from pocket_tts import voice_profiles
from pocket_tts.models.tts_model import TTSModel, export_model_state
from pocket_tts.utils.logging_utils import enable_logging
from pocket_tts.utils.utils import _ORIGINS_OF_PREDEFINED_VOICES

logger = logging.getLogger(__name__)

cli_app = typer.Typer(
    help="Kyutai Pocket TTS - Text-to-Speech generation tool", pretty_exceptions_show_locals=False
)


# ------------------------------------------------------
# The pocket-tts server implementation
# ------------------------------------------------------

# Global model instance
tts_model: TTSModel | None = None

# TTSModel.generate_audio/generate_audio_stream are explicitly not thread-safe
# (see tts_model.py docstrings). This serializes all model access across
# concurrent requests rather than letting them corrupt each other.
_generation_lock = threading.Lock()

# Set once at server startup via `serve --enable-pronunciation-check`.
_pronunciation_check_enabled = False

web_app = FastAPI(
    title="Kyutai Pocket TTS API", description="Text-to-Speech generation API", version="1.0.0"
)
web_app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "https://pod1-10007.internal.kyutai.org",
        "https://kyutai.org",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@web_app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the frontend."""
    static_path = Path(__file__).parent / "static" / "index.html"
    content = static_path.read_text()
    # Replace the placeholder with the actual default text prompt
    print(str(tts_model.origin))
    content = content.replace(
        "DEFAULT_TEXT_PROMPT", get_default_text_for_language(str(tts_model.origin))
    )
    return content


@web_app.get("/health")
async def health():
    return {"status": "healthy"}


@web_app.get("/profiles")
async def list_voice_profiles():
    return voice_profiles.list_profiles()


@web_app.get("/history")
async def get_history(profile: str | None = None, limit: int = 50):
    return history_module.list_history(profile_name=profile, limit=limit)


def _collect_chunks(chunks, collected: list):
    """Passthrough generator: yields chunks unchanged, also appending each one
    to `collected` so the full audio is available after streaming finishes."""
    for chunk in chunks:
        collected.append(chunk)
        yield chunk


def write_to_queue(queue, text_to_generate, model_state, profile_name, voice_source, lock):
    """Allows writing to the StreamingResponse as if it were a file."""

    class FileLikeToQueue(io.IOBase):
        def __init__(self, queue):
            self.queue = queue

        def write(self, data):
            self.queue.put(data)

        def flush(self):
            pass

        def close(self):
            self.queue.put(None)

    try:
        sample_rate = tts_model.config.mimi.sample_rate
        row_id_out: list = []
        collected_chunks: list | None = [] if _pronunciation_check_enabled else None

        audio_chunks = tts_model.generate_audio_stream(
            model_state=model_state, text_to_generate=text_to_generate
        )
        audio_chunks = history_module.track_and_log(
            audio_chunks,
            profile_name=profile_name,
            voice_source=voice_source,
            text=text_to_generate,
            source="server",
            sample_rate=sample_rate,
            row_id_out=row_id_out,
        )
        if collected_chunks is not None:
            audio_chunks = _collect_chunks(audio_chunks, collected_chunks)

        stream_audio_chunks(FileLikeToQueue(queue), audio_chunks, sample_rate)

        # Only after the response has fully streamed back: run the (slower)
        # pronunciation check in the background, never blocking the request.
        if collected_chunks:
            full_audio = torch.cat(collected_chunks, dim=0)
            threading.Thread(
                target=pronunciation_check.check_pronunciation,
                args=(full_audio, sample_rate, text_to_generate, row_id_out[0]),
                daemon=True,
            ).start()
    finally:
        lock.release()


def generate_data_with_state(
    text_to_generate: str,
    model_state: dict,
    profile_name: str | None,
    voice_source: str | None,
    lock,
):
    queue = Queue()

    # Run your function in a thread
    thread = threading.Thread(
        target=write_to_queue,
        args=(queue, text_to_generate, model_state, profile_name, voice_source, lock),
    )
    thread.start()

    # Yield data as it becomes available
    i = 0
    while True:
        data = queue.get()
        if data is None:
            break
        i += 1
        yield data

    thread.join()


@web_app.post("/tts")
def text_to_speech(
    text: str = Form(...),
    voice_url: str | None = Form(None),
    voice_wav: UploadFile | None = File(None),
    voice_profile: str | None = Form(None),
):
    """
    Generate speech from text using the pre-loaded voice prompt or a custom voice.

    Args:
        text: Text to convert to speech
        voice_url: Optional built-in voice name (e.g., "alba"), or voice URL (http://, https://, or hf://)
        voice_wav: Optional uploaded voice file (takes precedence over voice_url/voice_profile)
        voice_profile: Optional name of a saved voice profile (see `pocket-tts create-profile`)
    """
    if not text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")

    provided = [v for v in (voice_url, voice_wav, voice_profile) if v is not None]
    if len(provided) > 1:
        raise HTTPException(
            status_code=400,
            detail="Provide only one of voice_url, voice_wav, or voice_profile",
        )

    if not provided:
        voice_url = get_default_voice_for_language(str(tts_model.origin))

    # Model access begins here; released either below (on error) or by
    # write_to_queue once generation finishes (see _generation_lock comment above).
    _generation_lock.acquire()
    try:
        # Use the appropriate model state
        if voice_profile is not None:
            try:
                profile_path = voice_profiles.get_profile_path(voice_profile)
            except voice_profiles.ProfileNotFoundError as e:
                raise HTTPException(status_code=404, detail=str(e))
            text = voice_profiles.apply_rules(voice_profile, text)
            model_state = tts_model._cached_get_state_for_audio_prompt(str(profile_path))
            logging.warning("Using voice profile: %s", voice_profile)
        elif voice_url is not None:
            if not (
                voice_url.startswith("http://")
                or voice_url.startswith("https://")
                or voice_url.startswith("hf://")
                or voice_url in _ORIGINS_OF_PREDEFINED_VOICES
            ):
                raise HTTPException(
                    status_code=400,
                    detail="voice_url must start with http://, https://, or hf://",
                )
            model_state = tts_model._cached_get_state_for_audio_prompt(voice_url)
            logging.warning("Using voice from URL: %s", voice_url)
        elif voice_wav is not None:
            # Use uploaded voice file - preserve extension for format detection
            suffix = Path(voice_wav.filename).suffix if voice_wav.filename else ".wav"
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
                content = voice_wav.file.read()
                temp_file.write(content)
                temp_file.flush()
                temp_file_path = temp_file.name

            # Close the file before reading it back (required on Windows)
            try:
                model_state = tts_model.get_state_for_audio_prompt(
                    Path(temp_file_path), truncate=True
                )
            finally:
                os.unlink(temp_file_path)
        else:
            raise HTTPException(status_code=500, detail="This should never happen.")
    except Exception:
        _generation_lock.release()
        raise

    voice_source = voice_profile or voice_url or "upload"

    return StreamingResponse(
        generate_data_with_state(text, model_state, voice_profile, voice_source, _generation_lock),
        media_type="audio/wav",
        headers={
            "Content-Disposition": "attachment; filename=generated_speech.wav",
            "Transfer-Encoding": "chunked",
        },
    )


@cli_app.command()
def serve(
    host: Annotated[str, typer.Option(help="Host to bind to")] = "localhost",
    port: Annotated[int, typer.Option(help="Port to bind to")] = 8000,
    reload: Annotated[bool, typer.Option(help="Enable auto-reload")] = False,
    language: Annotated[
        str | None,
        typer.Option(
            help="Language for the TTS model. "
            "'english_2026-01', 'english_2026-04', 'english', 'french_24l', 'german_24l', 'portuguese', 'italian', 'spanish'."
            " Incompatible with the config argument. Default is 'english', which is the same model as 'english_2026-04'.",
            show_default=False,
        ),
    ] = None,
    config: Annotated[
        str | None,
        typer.Option(
            help="Path to locally-saved model config .yaml file. "
            "Incompatible with the language argument. If not provided, will use the default English model."
        ),
    ] = None,
    quantize: Annotated[
        bool, typer.Option(help="Apply int8 quantization to reduce memory usage")
    ] = False,
    enable_pronunciation_check: Annotated[
        bool,
        typer.Option(
            help="Transcribe generated audio in the background and flag mismatches "
            "with the requested text in /history. Requires `pip install 'pocket-tts[asr]'`. "
            "Runs after the response has already streamed back, so it never adds latency."
        ),
    ] = False,
):
    """Start the FastAPI server."""

    global tts_model, _pronunciation_check_enabled
    tts_model = TTSModel.load_model(language=language, config=config, quantize=quantize)

    _pronunciation_check_enabled = enable_pronunciation_check
    if enable_pronunciation_check:
        pronunciation_check.preload_model()

    uvicorn.run("pocket_tts.main:web_app", host=host, port=port, reload=reload)


# ------------------------------------------------------
# The pocket-tts single generation CLI implementation
# ------------------------------------------------------


@cli_app.command()
def generate(
    text: Annotated[str, typer.Option(help="Text to generate")] = None,
    voice: Annotated[
        str | None,
        typer.Option(
            help=(
                "Path to audio conditioning file (voice to clone). "
                "Defaults to a built-in voice chosen from the language: "
                "'giovanni' for italian, 'lola' for spanish, 'juergen' for german, "
                "'rafael' for portuguese, 'estelle' for french, 'alba' otherwise."
            ),
            show_default=False,
        ),
    ] = None,
    quiet: Annotated[bool, typer.Option("-q", "--quiet", help="Disable logging output")] = False,
    language: Annotated[
        str | None,
        typer.Option(
            help=(
                "Language for the TTS model. "
                "'english_2026-01', 'english_2026-04', 'english', 'french_24l', 'spanish_24l',"
                "'german_24l', 'portuguese_24l', 'italian_24l'."
                " Incompatible with the config argument. Default is 'english', which is the same model as 'english_2026-04'. "
                "The '24l' variants are bigger models, "
                "not distilled yet and here only as preview. They're not the final "
                "models for those languages."
            ),
            show_default=False,
        ),
    ] = None,
    config: Annotated[
        str | None,
        typer.Option(
            help="Path to locally-saved model config .yaml file. "
            "Incompatible with the language argument. If not provided, will use the default English model."
        ),
    ] = None,
    lsd_decode_steps: Annotated[
        int, typer.Option(help="Number of generation steps")
    ] = DEFAULT_LSD_DECODE_STEPS,
    temperature: Annotated[
        float, typer.Option(help="Temperature for generation")
    ] = DEFAULT_TEMPERATURE,
    noise_clamp: Annotated[float, typer.Option(help="Noise clamp value")] = DEFAULT_NOISE_CLAMP,
    eos_threshold: Annotated[float, typer.Option(help="EOS threshold")] = DEFAULT_EOS_THRESHOLD,
    frames_after_eos: Annotated[
        int, typer.Option(help="Number of frames to generate after EOS")
    ] = DEFAULT_FRAMES_AFTER_EOS,
    output_path: Annotated[
        str, typer.Option(help="Output path for generated audio")
    ] = "./tts_output.wav",
    device: Annotated[str, typer.Option(help="Device to use")] = "cpu",
    max_tokens: Annotated[
        int, typer.Option(help="Maximum number of tokens per chunk.")
    ] = MAX_TOKEN_PER_CHUNK,
    quantize: Annotated[
        bool, typer.Option(help="Apply int8 quantization to reduce memory usage")
    ] = False,
):
    """Generate speech using Kyutai Pocket TTS."""
    log_level = logging.ERROR if quiet else logging.INFO
    with enable_logging("pocket_tts", log_level):
        if text is None:
            text = get_default_text_for_language(language)
        if text == "-":
            # Read text from stdin
            text = sys.stdin.read()

        if not text.strip():
            logger.error("No input received from stdin.")
            raise typer.Exit(code=1)
        tts_model = TTSModel.load_model(
            language=language,
            config=config,
            temp=temperature,
            lsd_decode_steps=lsd_decode_steps,
            noise_clamp=noise_clamp,
            eos_threshold=eos_threshold,
            quantize=quantize,
        )
        tts_model.to(device)

        if voice is None:
            voice = get_default_voice_for_language(language)
        model_state_for_voice = tts_model.get_state_for_audio_prompt(voice)
        # Stream audio generation directly to file or stdout
        audio_chunks = tts_model.generate_audio_stream(
            model_state=model_state_for_voice,
            text_to_generate=text,
            frames_after_eos=frames_after_eos,
            max_tokens=max_tokens,
        )
        audio_chunks = history_module.track_and_log(
            audio_chunks,
            profile_name=None,
            voice_source=str(voice),
            text=text,
            source="cli",
            sample_rate=tts_model.config.mimi.sample_rate,
        )

        stream_audio_chunks(output_path, audio_chunks, tts_model.config.mimi.sample_rate)

        # Only print the result message if not writing to stdout
        if output_path != "-":
            logger.info("Results written in %s", output_path)
        logger.info("-" * 20)
        logger.info(
            "If you want to try multiple voices and prompts quickly, try the `serve` command."
        )
        logger.info(
            "If you like Kyutai projects, comment, like, subscribe at https://x.com/kyutai_labs"
        )


# ----------------------------------------------
# export audio to safetensors CLI implementation
# ----------------------------------------------


@cli_app.command()
def export_voice(
    audio_path: Annotated[
        str, typer.Argument(help="Audio file or directory to convert and export")
    ],
    export_path: Annotated[str, typer.Argument(help="Output file or directory")],
    quiet: Annotated[bool, typer.Option("-q", "--quiet", help="Disable logging output")] = False,
    language: Annotated[
        str | None,
        typer.Option(
            help=(
                "Language for the TTS model. "
                "'english_2026-01', 'english_2026-04', 'english', 'french_24l', 'german_24l','spanish_24l',"
                " 'portuguese_24l', 'italian_24l'."
                " Incompatible with the config argument. Default is 'english', which is the same model as 'english_2026-04'. "
                "The '24l' variants are bigger models, "
                "not distilled yet and here only as preview."
            ),
            show_default=False,
        ),
    ] = None,
    config: Annotated[
        str | None,
        typer.Option(
            help="Path to locally-saved model config .yaml file. "
            "Incompatible with the language argument. If not provided, will use the default English model."
        ),
    ] = None,
):
    """Convert and save audio to .safetensors file"""

    log_level = logging.ERROR if quiet else logging.INFO
    with enable_logging("pocket_tts", log_level):
        tts_model = TTSModel.load_model(language=language, config=config)
        model_state = tts_model.get_state_for_audio_prompt(
            audio_conditioning=audio_path, truncate=True
        )
        export_model_state(model_state, export_path)


# ----------------------------------------------
# voice profiles CLI implementation
# ----------------------------------------------


@cli_app.command("create-profile")
def create_profile(
    audio_path: Annotated[str, typer.Argument(help="Audio file to build the profile from")],
    name: Annotated[str, typer.Argument(help="Name for the profile (letters, digits, _, -)")],
    quiet: Annotated[bool, typer.Option("-q", "--quiet", help="Disable logging output")] = False,
    language: Annotated[
        str | None,
        typer.Option(
            help="Language for the TTS model used to encode the voice.", show_default=False
        ),
    ] = None,
    config: Annotated[
        str | None,
        typer.Option(
            help="Path to locally-saved model config .yaml file. "
            "Incompatible with the language argument."
        ),
    ] = None,
    tags: Annotated[
        str | None, typer.Option(help="Comma-separated tags, e.g. 'medical,calm'")
    ] = None,
    notes: Annotated[str | None, typer.Option(help="Free-form notes about this voice")] = None,
    overwrite: Annotated[
        bool, typer.Option(help="Replace an existing profile with the same name")
    ] = False,
):
    """Create a reusable, named voice profile from an audio file."""
    log_level = logging.ERROR if quiet else logging.INFO
    with enable_logging("pocket_tts", log_level):
        tts_model = TTSModel.load_model(language=language, config=config)
        model_state = tts_model.get_state_for_audio_prompt(audio_path, truncate=True)
        path = voice_profiles.save_profile(
            name,
            model_state,
            source=audio_path,
            language=language,
            tags=[t.strip() for t in tags.split(",")] if tags else [],
            notes=notes or "",
            overwrite=overwrite,
        )
        logger.info("Saved profile '%s' to %s", name, path)


@cli_app.command("list-profiles")
def list_profiles_command():
    """List saved voice profiles."""
    profiles = voice_profiles.list_profiles()
    if not profiles:
        typer.echo("No voice profiles saved yet. Create one with `pocket-tts create-profile`.")
        return
    for p in profiles:
        typer.echo(
            f"{p['name']:20s} lang={p.get('language') or '-':10s} tags={','.join(p['tags'])}"
        )


# ----------------------------------------------
# voice profile rules CLI implementation
# ----------------------------------------------


@cli_app.command("add-rule")
def add_rule(
    profile: Annotated[str, typer.Argument(help="Name of the voice profile")],
    pattern: Annotated[str, typer.Argument(help="Text to match")],
    replacement: Annotated[str, typer.Argument(help="Text to substitute in")],
    regex: Annotated[
        bool, typer.Option(help="Treat pattern as a regex instead of a literal word")
    ] = False,
):
    """Add a text-substitution rule to a saved voice profile."""
    voice_profiles.add_rule(profile, pattern, replacement, regex=regex)
    typer.echo(f"Added rule to '{profile}': {pattern!r} -> {replacement!r}")


@cli_app.command("list-rules")
def list_rules_command(
    profile: Annotated[str, typer.Argument(help="Name of the voice profile")],
):
    """List text-substitution rules for a voice profile."""
    rules = voice_profiles.list_rules(profile)
    if not rules:
        typer.echo(f"No rules for '{profile}' yet. Add one with `pocket-tts add-rule`.")
        return
    for i, r in enumerate(rules):
        typer.echo(f"[{i}] {r['pattern']!r} -> {r['replacement']!r}  (regex={r['regex']})")


@cli_app.command("remove-rule")
def remove_rule_command(
    profile: Annotated[str, typer.Argument(help="Name of the voice profile")],
    index: Annotated[int, typer.Argument(help="Rule index, see list-rules")],
):
    """Remove a rule from a voice profile by its index."""
    voice_profiles.remove_rule(profile, index)
    typer.echo(f"Removed rule [{index}] from '{profile}'")


@cli_app.command("import-rules")
def import_rules(
    profile: Annotated[str, typer.Argument(help="Name of the voice profile")],
    rules_file: Annotated[
        str, typer.Argument(help="Path to a JSON file: a list of {pattern, replacement, regex?}")
    ],
):
    """Bulk-add rules to a profile from a JSON file."""
    rules = json.loads(Path(rules_file).read_text())
    count = voice_profiles.add_rules(profile, rules)
    typer.echo(f"Added {count} rule(s) to '{profile}'")


@cli_app.command("remove-rules")
def remove_rules_command(
    profile: Annotated[str, typer.Argument(help="Name of the voice profile")],
    indices: Annotated[list[int], typer.Argument(help="Rule indices, see list-rules")],
):
    """Remove multiple rules from a profile by index."""
    voice_profiles.remove_rules(profile, indices)
    typer.echo(f"Removed {len(indices)} rule(s) from '{profile}'")


@cli_app.command("clear-rules")
def clear_rules_command(
    profile: Annotated[str, typer.Argument(help="Name of the voice profile")],
):
    """Remove all rules from a profile."""
    voice_profiles.clear_rules(profile)
    typer.echo(f"Cleared all rules from '{profile}'")


# ----------------------------------------------
# generation history CLI implementation
# ----------------------------------------------


@cli_app.command()
def history(
    profile: Annotated[str | None, typer.Option(help="Filter by profile name")] = None,
    limit: Annotated[int, typer.Option(help="Max rows to show")] = 20,
):
    """Show recent generation history."""
    rows = history_module.list_history(profile_name=profile, limit=limit)
    if not rows:
        typer.echo("No generation history yet.")
        return
    for row in rows:
        snippet = row["text"][:60] + ("..." if len(row["text"]) > 60 else "")
        typer.echo(
            f"{row['created_at']}  {row['source']:6s}  voice={row['voice_source'] or '-':20s}  \"{snippet}\""
        )


if __name__ == "__main__":
    cli_app()
