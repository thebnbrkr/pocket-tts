import json
import re
import uuid
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
        "id": str(uuid.uuid4()),
        "name": name,
        "source": source,
        "language": language,
        "tags": tags or [],
        "notes": notes,
        "rules": [],
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2))
    return safetensors_path


def _load_metadata(name: str) -> dict:
    _validate_name(name)
    metadata_path = _profiles_dir() / f"{name}.json"
    if not metadata_path.exists():
        raise ProfileNotFoundError(f"Unknown voice profile: {name}")
    return json.loads(metadata_path.read_text())


def _save_metadata(name: str, metadata: dict) -> None:
    metadata_path = _profiles_dir() / f"{name}.json"
    metadata_path.write_text(json.dumps(metadata, indent=2))


def add_rule(name: str, pattern: str, replacement: str, regex: bool = False) -> None:
    metadata = _load_metadata(name)
    metadata.setdefault("rules", []).append(
        {"pattern": pattern, "replacement": replacement, "regex": regex}
    )
    _save_metadata(name, metadata)


def remove_rule(name: str, index: int) -> None:
    metadata = _load_metadata(name)
    rules = metadata.setdefault("rules", [])
    rules.pop(index)
    _save_metadata(name, metadata)


def add_rules(name: str, rules: list[dict]) -> int:
    for i, rule in enumerate(rules):
        if "pattern" not in rule or "replacement" not in rule:
            raise ValueError(f"Rule at index {i} is missing 'pattern' or 'replacement': {rule}")

    metadata = _load_metadata(name)
    existing = metadata.setdefault("rules", [])
    for rule in rules:
        existing.append(
            {
                "pattern": rule["pattern"],
                "replacement": rule["replacement"],
                "regex": rule.get("regex", False),
            }
        )
    _save_metadata(name, metadata)
    return len(rules)


def remove_rules(name: str, indices: list[int]) -> None:
    metadata = _load_metadata(name)
    rules = metadata.setdefault("rules", [])
    for index in sorted(set(indices), reverse=True):
        rules.pop(index)
    _save_metadata(name, metadata)


def clear_rules(name: str) -> None:
    metadata = _load_metadata(name)
    metadata["rules"] = []
    _save_metadata(name, metadata)


def list_rules(name: str) -> list[dict]:
    return _load_metadata(name).get("rules", [])


def apply_rules(name: str, text: str) -> str:
    for rule in list_rules(name):
        if rule["regex"]:
            text = re.sub(rule["pattern"], rule["replacement"], text)
        else:
            text = re.sub(
                rf"\b{re.escape(rule['pattern'])}\b",
                rule["replacement"],
                text,
                flags=re.IGNORECASE,
            )
    return text


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
