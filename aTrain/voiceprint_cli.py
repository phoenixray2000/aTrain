from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from aTrain_core.load_resources import get_model

from aTrain.voiceprint_identification import extract_embedding
from aTrain.voiceprints import (
    EMBEDDING_MODEL_ID,
    VOICEPRINT_SCHEMA_VERSION,
    VoiceprintProfile,
    load_voiceprint,
    merge_centroid,
    save_voiceprint,
    validate_voiceprint_name,
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _normalise_vector(vector: np.ndarray) -> np.ndarray:
    array = np.asarray(vector, dtype=np.float32)
    if array.ndim != 1 or array.size == 0:
        raise ValueError("Voiceprint embedding must be a non-empty 1-D vector.")
    norm = float(np.linalg.norm(array))
    if norm <= 0:
        raise ValueError("Voiceprint embedding cannot be a zero vector.")
    return array / norm


def read_embedding_file(path: Path) -> tuple[list[str], np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(f"Speaker embedding file does not exist: {path}")
    with np.load(path, allow_pickle=False) as payload:
        labels = [str(item) for item in payload["labels"].tolist()]
        embeddings = np.asarray(payload["embeddings"], dtype=np.float32)
    if embeddings.ndim != 2 or len(labels) != embeddings.shape[0]:
        raise ValueError("Speaker embedding file is invalid: labels and embeddings do not match.")
    return labels, embeddings


def _save_new_or_update(
    name: str, embedding: np.ndarray, enrollment: dict[str, Any], update: bool
) -> VoiceprintProfile:
    cleaned_name = validate_voiceprint_name(name)
    vector = _normalise_vector(embedding)
    try:
        existing = load_voiceprint(cleaned_name)
    except FileNotFoundError:
        profile = VoiceprintProfile(
            name=cleaned_name,
            model_id=EMBEDDING_MODEL_ID,
            schema_version=VOICEPRINT_SCHEMA_VERSION,
            embedding_dim=int(vector.shape[0]),
            embedding=vector,
            enrollments=[enrollment],
        )
        save_voiceprint(profile)
        return profile

    if not update:
        raise FileExistsError(f"Voiceprint already exists: {cleaned_name}. Use --update to merge a new sample.")
    merged = merge_centroid(existing.embedding, vector, max(1, len(existing.enrollments)))
    profile = VoiceprintProfile(
        name=existing.name,
        model_id=existing.model_id,
        schema_version=existing.schema_version,
        embedding_dim=existing.embedding_dim,
        embedding=merged,
        enrollments=[*existing.enrollments, enrollment],
    )
    save_voiceprint(profile)
    return profile


def enroll_voiceprint_from_audio(
    audio_path: Path,
    name: str,
    update: bool,
    device,
    min_duration_sec: float = 3.0,
) -> VoiceprintProfile:
    model_path = get_model("speaker-detection")
    embedding = extract_embedding(audio_path, model_path, device, min_duration_sec=min_duration_sec)
    return _save_new_or_update(
        name,
        embedding,
        {
            "source_type": "audio",
            "source_path": str(audio_path),
            "created_at": _utc_now(),
        },
        update,
    )


def enroll_voiceprint_from_speaker_embedding(
    embedding_file: Path,
    speaker_label: str,
    name: str,
    update: bool,
    source: str | None = None,
) -> VoiceprintProfile:
    labels, embeddings = read_embedding_file(embedding_file)
    try:
        index = labels.index(speaker_label)
    except ValueError as error:
        raise ValueError(f"Speaker label not found in embedding file: {speaker_label}") from error
    return _save_new_or_update(
        name,
        embeddings[index],
        {
            "source_type": "speaker_embedding",
            "source_path": str(embedding_file),
            "speaker_label": speaker_label,
            "source": source,
            "created_at": _utc_now(),
        },
        update,
    )
