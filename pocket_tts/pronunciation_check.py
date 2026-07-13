import logging
import re
import tempfile
from pathlib import Path

import scipy.io.wavfile
import torch

from pocket_tts import history

logger = logging.getLogger(__name__)

_asr_model = None

ASR_NOT_INSTALLED = (
    "Pronunciation checking requires faster-whisper, which is not installed. "
    "Install it with: pip install 'pocket-tts[asr]'"
)


def _get_asr_model():
    global _asr_model
    if _asr_model is None:
        try:
            from faster_whisper import WhisperModel
        except ImportError as e:
            raise ImportError(ASR_NOT_INSTALLED) from e
        logger.info("Loading pronunciation-check ASR model (faster-whisper, tiny)...")
        _asr_model = WhisperModel("tiny", device="cpu", compute_type="int8")
    return _asr_model


def preload_model() -> None:
    """Load the ASR model eagerly (called at server startup) so a missing
    dependency fails loudly and immediately, not silently inside a background
    thread on the first request."""
    _get_asr_model()


def _normalize(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    return " ".join(text.split())


def check_pronunciation(audio: torch.Tensor, sample_rate: int, intended_text: str, row_id: int) -> None:
    """Transcribe the generated audio and compare it to the text that was
    asked for, updating the corresponding history row. Runs in a background
    thread after the response has already been sent -- never raises out to
    the caller, only logs, since nothing is waiting on this result directly.
    """
    try:
        model = _get_asr_model()

        audio_int16 = (audio.clamp(-1, 1) * 32767).short().numpy()
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            temp_path = f.name
        try:
            scipy.io.wavfile.write(temp_path, sample_rate, audio_int16)
            segments, _ = model.transcribe(temp_path)
            transcribed_text = " ".join(segment.text for segment in segments).strip()
        finally:
            Path(temp_path).unlink(missing_ok=True)

        matched = _normalize(transcribed_text) == _normalize(intended_text)
        history.update_transcription(row_id, transcribed_text, matched)
    except Exception:
        logger.exception("Pronunciation check failed for history row %d", row_id)
