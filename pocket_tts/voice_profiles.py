import json
import re
from datetime import datetime, timezone
from pathlib import Path

from pocket_tts.models.tts_model import export_model_state
from pocket_tts.utils.utils import make_cache_directory

_NAME_RE = re.compile(r"^[A-Za-z0-9_-]+$")


class ProfileNotFoundError(Exception):
    pass


def _profiles_dir() -> Path:
    profiles_dir = make_cache_directory() / "profiles"
    profiles_dir.mkdir(parents=True, exist_ok=True)
    return profiles_dir


def _validate_name(name: str) -> None:
    if not _NAME_RE.match(name):
        raise ValueError(
            f"Invalid profile name {name!r}: only letters, digits, '_' and '-' are allowed."
        )


def save_profile(
    name: str,
    model_state: dict,
    *,
    source: str,
    language: str | None = None,
    tags: list[str] | None = None,
    notes: str = "",
    overwrite: bool = False,
) -> Path:
    _validate_name(name)
    profiles_dir = _profiles_dir()
    safetensors_path = profiles_dir / f"{name}.safetensors"
    metadata_path = profiles_dir / f"{name}.json"

    if not overwrite and (safetensors_path.exists() or metadata_path.exists()):
        raise FileExistsError(
            f"Profile '{name}' already exists. Pass overwrite=True to replace it."
        )

    export_model_state(model_state, safetensors_path)
    metadata = {
        "name": name,
        "source": source,
        "language": language,
        "tags": tags or [],
        "notes": notes,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2))
    return safetensors_path


def get_profile_path(name: str) -> Path:
    _validate_name(name)
    safetensors_path = _profiles_dir() / f"{name}.safetensors"
    if not safetensors_path.exists():
        raise ProfileNotFoundError(f"Unknown voice profile: {name}")
    return safetensors_path


def list_profiles() -> list[dict]:
    profiles = []
    for metadata_path in sorted(_profiles_dir().glob("*.json")):
        profiles.append(json.loads(metadata_path.read_text()))
    return profiles


def delete_profile(name: str) -> None:
    _validate_name(name)
    profiles_dir = _profiles_dir()
    (profiles_dir / f"{name}.safetensors").unlink(missing_ok=True)
    (profiles_dir / f"{name}.json").unlink(missing_ok=True)
