# aTrain CLI Voiceprint Enrollment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add aTrain CLI support for exporting diarization speaker embeddings and enrolling/updating local voiceprint profiles from either an exported speaker embedding or a direct audio sample.

**Architecture:** Reuse the existing voiceprint storage and matching code in `aTrain/voiceprints.py` and the existing pyannote embedding extraction/capture code in `aTrain/voiceprint_identification.py`. The CLI adds a focused `voiceprint enroll` subcommand and a `transcribe --speaker-embeddings-output` option; no GUI or collab-runtime code changes live in this plan. The collab-runtime Workbench will learn from the per-speaker embedding artifact exported during transcription, while manual CLI users may also enroll from a standalone audio sample.

**Tech Stack:** Python, Typer, unittest, NumPy, existing `aTrain_core` model loading, existing `aTrain` voiceprint JSON schema, Windows PowerShell commands.

---

## Current Evidence

- `D:\Git\aTrain\aTrain\cli.py` defines `cli = typer.Typer(...)`, `transcribe`, `_transcribe_one`, `_run_batch`, and `_identify_captured_speakers`.
- `transcribe` already supports `--speaker-detection`, `--identify-speakers`, `--voiceprint-threshold`, and `--voiceprint-margin`.
- `_transcribe_one()` stages a transcription under a temporary `TRANSCRIPT_DIR`, calls `patch_core_speaker_capture()` when speaker identification is enabled, reads captured speaker embeddings with `read_captured_embeddings()`, then deletes the staging directory.
- `D:\Git\aTrain\aTrain\voiceprint_identification.py` defines `CAPTURE_FILENAME = "_speaker_embeddings.npz"`, `extract_embedding()`, `read_captured_embeddings()`, and `_write_captured_embeddings()`.
- `D:\Git\aTrain\aTrain\voiceprints.py` defines `VoiceprintProfile`, `validate_voiceprint_name()`, `load_voiceprint()`, `save_voiceprint()`, `list_voiceprints()`, `remove_voiceprint()`, `merge_centroid()`, `cosine_similarity_matrix()`, and `assign_voiceprints()`.
- `D:\Git\aTrain\aTrain\utils\voiceprints.py` already has GUI enrollment logic and proves the storage layer supports profile creation and update.
- `D:\Git\aTrain\README.md` currently says the CLI consumes but does not create/delete voiceprints; this becomes stale after this feature.
- Existing tests include `tests/test_voiceprint_identification.py`, `tests/test_voiceprints.py`, `tests/test_cli_voiceprints.py`, and `tests/test_gui_voiceprint_identification.py`.

## CLI Contract

Export captured per-speaker embeddings during transcription:

```powershell
aTrain-cli transcribe "D:\input\meeting.wav" --speaker-detection --identify-speakers --speaker-embeddings-output "D:\out\meeting.speaker-embeddings.npz"
```

Enroll/update from an exported speaker embedding:

```powershell
aTrain-cli voiceprint enroll --name "李想" --speaker-embeddings "D:\out\meeting.speaker-embeddings.npz" --speaker SPEAKER_01 --update
```

Enroll/update from a direct audio sample:

```powershell
aTrain-cli voiceprint enroll --name "李想" --audio "D:\samples\li-xiang.wav" --update --device CPU --min-duration-sec 3
```

Validation rules:
- exactly one of `--audio` and `--speaker-embeddings` is required;
- `--speaker` is required with `--speaker-embeddings`;
- `--speaker` is rejected with `--audio`;
- existing profile without `--update` raises a clear error;
- `--update` merges the new vector into the existing centroid with `merge_centroid()`;
- invalid names reuse `validate_voiceprint_name()`;
- low-duration or undecodable audio errors are surfaced from `extract_embedding()` without swallowing the original message;
- if transcription did not capture speaker embeddings, `--speaker-embeddings-output` fails with a clear non-zero CLI error instead of writing an empty file.

## Source Of Learning Audio

There are two learning sources:

1. Workbench/collab-runtime source: the current recording's diarization speaker embedding exported by `aTrain-cli transcribe --speaker-embeddings-output`. This is the first implementation path because it avoids inventing a separate audio clipping pipeline in collab-runtime.
2. Manual CLI source: a user-provided audio sample path via `--audio`. This is useful for standalone aTrain operation and regression tests.

## File Structure

- Create `D:\Git\aTrain\aTrain\voiceprint_cli.py`: reusable enrollment helpers for audio samples and captured speaker embeddings.
- Modify `D:\Git\aTrain\aTrain\cli.py`: add `voiceprint` Typer sub-app, `voiceprint enroll` command, `--speaker-embeddings-output` transcribe option, and copy captured embeddings before staging cleanup.
- Modify `D:\Git\aTrain\aTrain\voiceprint_identification.py` only if a helper is needed to resolve/copy the captured embedding artifact cleanly.
- Create `D:\Git\aTrain\tests\test_voiceprint_cli.py`: unit tests for helper validation, creation, update, and captured-speaker selection.
- Modify `D:\Git\aTrain\tests\test_cli_voiceprints.py`: tests for the new Typer subcommand and `transcribe --speaker-embeddings-output` option.
- Modify `D:\Git\aTrain\README.md`: document CLI enrollment and remove the stale “CLI does not create voiceprints” statement.

---

### Task 1: Add Voiceprint Enrollment Helpers

**Files:**
- Create: `D:\Git\aTrain\aTrain\voiceprint_cli.py`
- Create: `D:\Git\aTrain\tests\test_voiceprint_cli.py`

- [ ] **Step 1: Write failing helper tests**

Create `tests/test_voiceprint_cli.py` with tests that patch model loading and storage functions:

```python
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from aTrain.voiceprints import VoiceprintProfile, VOICEPRINT_SCHEMA_VERSION, EMBEDDING_MODEL_ID


class VoiceprintCliTests(unittest.TestCase):
    def test_audio_enrollment_creates_new_profile(self):
        from aTrain.voiceprint_cli import enroll_voiceprint_from_audio

        vector = np.array([1.0, 0.0], dtype=np.float32)
        with patch("aTrain.voiceprint_cli.extract_embedding", return_value=vector), \
            patch("aTrain.voiceprint_cli.load_voiceprint", side_effect=FileNotFoundError("missing")), \
            patch("aTrain.voiceprint_cli.save_voiceprint") as save:
            profile = enroll_voiceprint_from_audio(Path("sample.wav"), "李想", update=False, device="CPU")

        self.assertEqual(profile.name, "李想")
        self.assertEqual(profile.schema_version, VOICEPRINT_SCHEMA_VERSION)
        self.assertEqual(profile.model_id, EMBEDDING_MODEL_ID)
        self.assertEqual(profile.embedding_dim, 2)
        self.assertEqual(profile.enrollments[0]["source_type"], "audio")
        save.assert_called_once()

    def test_existing_profile_without_update_fails(self):
        from aTrain.voiceprint_cli import enroll_voiceprint_from_audio

        existing = VoiceprintProfile(
            name="李想",
            model_id=EMBEDDING_MODEL_ID,
            schema_version=VOICEPRINT_SCHEMA_VERSION,
            embedding_dim=2,
            embedding=np.array([1.0, 0.0], dtype=np.float32),
            enrollments=[],
        )
        with patch("aTrain.voiceprint_cli.extract_embedding", return_value=np.array([0.0, 1.0], dtype=np.float32)), \
            patch("aTrain.voiceprint_cli.load_voiceprint", return_value=existing):
            with self.assertRaises(FileExistsError):
                enroll_voiceprint_from_audio(Path("sample.wav"), "李想", update=False, device="CPU")

    def test_existing_profile_with_update_merges_centroid(self):
        from aTrain.voiceprint_cli import enroll_voiceprint_from_audio

        existing = VoiceprintProfile(
            name="李想",
            model_id=EMBEDDING_MODEL_ID,
            schema_version=VOICEPRINT_SCHEMA_VERSION,
            embedding_dim=2,
            embedding=np.array([1.0, 0.0], dtype=np.float32),
            enrollments=[{"source_type": "audio"}],
        )
        with patch("aTrain.voiceprint_cli.extract_embedding", return_value=np.array([0.0, 1.0], dtype=np.float32)), \
            patch("aTrain.voiceprint_cli.load_voiceprint", return_value=existing), \
            patch("aTrain.voiceprint_cli.save_voiceprint") as save:
            profile = enroll_voiceprint_from_audio(Path("sample.wav"), "李想", update=True, device="CPU")

        np.testing.assert_allclose(profile.embedding, np.array([0.5, 0.5], dtype=np.float32))
        self.assertEqual(len(profile.enrollments), 2)
        save.assert_called_once()

    def test_speaker_embedding_enrollment_selects_requested_label(self):
        from aTrain.voiceprint_cli import enroll_voiceprint_from_speaker_embedding

        labels = ["SPEAKER_00", "SPEAKER_01"]
        embeddings = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        with patch("aTrain.voiceprint_cli.read_embedding_file", return_value=(labels, embeddings)), \
            patch("aTrain.voiceprint_cli.load_voiceprint", side_effect=FileNotFoundError("missing")), \
            patch("aTrain.voiceprint_cli.save_voiceprint"):
            profile = enroll_voiceprint_from_speaker_embedding(Path("speakers.npz"), "SPEAKER_01", "李想", update=False)

        np.testing.assert_allclose(profile.embedding, np.array([0.0, 1.0], dtype=np.float32))
        self.assertEqual(profile.enrollments[0]["speaker_label"], "SPEAKER_01")

    def test_speaker_embedding_enrollment_rejects_missing_label(self):
        from aTrain.voiceprint_cli import enroll_voiceprint_from_speaker_embedding

        with patch("aTrain.voiceprint_cli.read_embedding_file", return_value=(["SPEAKER_00"], np.array([[1.0, 0.0]], dtype=np.float32))):
            with self.assertRaisesRegex(ValueError, "Speaker label not found"):
                enroll_voiceprint_from_speaker_embedding(Path("speakers.npz"), "SPEAKER_01", "李想", update=False)


if __name__ == "__main__":
    unittest.main()
```

Run:

```powershell
cd D:\Git\aTrain
.\.venv\Scripts\python.exe -m unittest tests.test_voiceprint_cli -v
```

Expected before implementation: import failure for `aTrain.voiceprint_cli`.

- [ ] **Step 2: Implement helper module**

Create `aTrain/voiceprint_cli.py` with these functions:

```python
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

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


def _save_new_or_update(name: str, embedding: np.ndarray, enrollment: dict[str, Any], update: bool) -> VoiceprintProfile:
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
    embedding = extract_embedding(audio_path, None, device, min_duration_sec=min_duration_sec)
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
```

If `extract_embedding()` requires a real model path instead of `None`, resolve the model path using the same project helper used by the GUI enrollment path before calling it.

- [ ] **Step 3: Verify helper tests pass**

Run:

```powershell
cd D:\Git\aTrain
.\.venv\Scripts\python.exe -m unittest tests.test_voiceprint_cli -v
```

Expected after implementation: all tests in `tests.test_voiceprint_cli` pass.

- [ ] **Step 4: Commit helper task**

Run:

```powershell
cd D:\Git\aTrain
git status -sb
git add aTrain\voiceprint_cli.py tests\test_voiceprint_cli.py
git commit -m "Add voiceprint enrollment helpers"
```

Expected: commit contains only the helper module and helper tests.

---

### Task 2: Add `aTrain-cli voiceprint enroll`

**Files:**
- Modify: `D:\Git\aTrain\aTrain\cli.py`
- Modify: `D:\Git\aTrain\tests\test_cli_voiceprints.py`

- [ ] **Step 1: Add failing Typer command tests**

In the existing `CliVoiceprintTests(unittest.TestCase)` class in `tests/test_cli_voiceprints.py`, add tests with `typer.testing.CliRunner` and `unittest.mock`:

```python
def test_voiceprint_enroll_rejects_missing_source(self):
    runner = CliRunner()

    result = runner.invoke(cli, ["voiceprint", "enroll", "--name", "李想"])

    self.assertEqual(result.exit_code, 2)
    self.assertIn("Provide exactly one of --audio or --speaker-embeddings", result.stderr)


def test_voiceprint_enroll_rejects_both_sources(self):
    runner = CliRunner()

    result = runner.invoke(
        cli,
        [
            "voiceprint",
            "enroll",
            "--name",
            "李想",
            "--audio",
            "sample.wav",
            "--speaker-embeddings",
            "speakers.npz",
            "--speaker",
            "SPEAKER_01",
        ],
    )

    self.assertEqual(result.exit_code, 2)
    self.assertIn("Provide exactly one of --audio or --speaker-embeddings", result.stderr)


def test_voiceprint_enroll_from_speaker_embedding_calls_helper(self):
    runner = CliRunner()
    called = {}

    def fake_enroll(path, speaker, name, update, source=None):
        called.update(path=path, speaker=speaker, name=name, update=update, source=source)

    with tempfile.TemporaryDirectory() as temp_dir:
        embedding_file = Path(temp_dir) / "speakers.npz"
        embedding_file.write_bytes(b"npz-stub")
        with mock.patch("aTrain.cli.enroll_voiceprint_from_speaker_embedding", side_effect=fake_enroll):
            result = runner.invoke(
                cli,
                [
                    "voiceprint",
                    "enroll",
                    "--name",
                    "李想",
                    "--speaker-embeddings",
                    str(embedding_file),
                    "--speaker",
                    "SPEAKER_01",
                    "--update",
                    "--source",
                    "recording:abc",
                ],
            )

    self.assertEqual(result.exit_code, 0)
    self.assertEqual(called["speaker"], "SPEAKER_01")
    self.assertEqual(called["name"], "李想")
    self.assertIs(called["update"], True)
```

Run:

```powershell
cd D:\Git\aTrain
.\.venv\Scripts\python.exe -m unittest tests.test_cli_voiceprints -v
```

Expected before implementation: `No such command 'voiceprint'` or import errors.

- [ ] **Step 2: Wire Typer sub-app**

In `aTrain/cli.py`, import helpers:

```python
from aTrain.voiceprint_cli import (
    enroll_voiceprint_from_audio,
    enroll_voiceprint_from_speaker_embedding,
)
```

After `cli = typer.Typer(...)`, add:

```python
voiceprint_cli = typer.Typer(help="Manage speaker voiceprints.", no_args_is_help=True)
cli.add_typer(voiceprint_cli, name="voiceprint")
```

Add command:

```python
@voiceprint_cli.command("enroll")
def voiceprint_enroll(
    name: Annotated[str, typer.Option("--name", help="Person name for the voiceprint.")],
    audio: Annotated[Path | None, typer.Option("--audio", help="Audio sample used for enrollment.")] = None,
    speaker_embeddings: Annotated[
        Path | None,
        typer.Option("--speaker-embeddings", help="NPZ speaker embedding artifact exported by transcribe."),
    ] = None,
    speaker: Annotated[str | None, typer.Option("--speaker", help="Speaker label to enroll, such as SPEAKER_01.")] = None,
    update: Annotated[bool, typer.Option("--update", help="Merge into an existing profile.")] = False,
    source: Annotated[str | None, typer.Option("--source", help="Optional audit source stored with the enrollment.")] = None,
    device: Annotated[Device, typer.Option(help="Hardware used for audio embedding extraction.")] = Device.CPU,
    min_duration_sec: Annotated[
        float,
        typer.Option("--min-duration-sec", help="Minimum audio duration accepted for direct audio enrollment.", min=0.1),
    ] = 3.0,
):
    try:
        source_count = int(audio is not None) + int(speaker_embeddings is not None)
        if source_count != 1:
            raise ValueError("Provide exactly one of --audio or --speaker-embeddings.")
        if speaker_embeddings is not None and not speaker:
            raise ValueError("--speaker is required with --speaker-embeddings.")
        if audio is not None and speaker:
            raise ValueError("--speaker can only be used with --speaker-embeddings.")
        if audio is not None:
            profile = enroll_voiceprint_from_audio(audio, name, update=update, device=device, min_duration_sec=min_duration_sec)
        else:
            profile = enroll_voiceprint_from_speaker_embedding(speaker_embeddings, speaker, name, update=update, source=source)
    except Exception as error:
        typer.echo(str(error), err=True)
        raise typer.Exit(code=2) from error

    typer.echo(f"Voiceprint enrolled: {profile.name}")
```

- [ ] **Step 3: Verify Typer command tests**

Run:

```powershell
cd D:\Git\aTrain
.\.venv\Scripts\python.exe -m unittest tests.test_cli_voiceprints -v
```

Expected after implementation: existing CLI voiceprint tests and new `voiceprint enroll` command tests pass.

- [ ] **Step 4: Commit command task**

Run:

```powershell
cd D:\Git\aTrain
git status -sb
git add aTrain\cli.py tests\test_cli_voiceprints.py
git commit -m "Add voiceprint enroll CLI command"
```

Expected: commit contains only CLI command wiring and command tests.

---

### Task 3: Export Captured Speaker Embeddings From Transcription

**Files:**
- Modify: `D:\Git\aTrain\aTrain\cli.py`
- Modify: `D:\Git\aTrain\aTrain\voiceprint_identification.py` only if a small exported helper is needed
- Modify: `D:\Git\aTrain\tests\test_cli_voiceprints.py`

- [ ] **Step 1: Add failing export tests**

In the existing `CliVoiceprintTests(unittest.TestCase)` class in `tests/test_cli_voiceprints.py`, add focused helper tests for copying the captured speaker embedding artifact:

```python
def test_copy_speaker_embeddings_writes_requested_output(self):
    from aTrain import cli as cli_module

    with tempfile.TemporaryDirectory() as temp_dir:
        root = Path(temp_dir)
        staging_dir = root / "staging"
        file_dir = staging_dir / "file-id"
        file_dir.mkdir(parents=True)
        captured = file_dir / "_speaker_embeddings.npz"
        captured.write_bytes(b"npz-data")
        output_path = root / "exported.npz"

        result = cli_module._copy_speaker_embeddings(staging_dir, "file-id", output_path)

        self.assertEqual(output_path.read_bytes(), b"npz-data")
        self.assertEqual(result, output_path)


def test_copy_speaker_embeddings_fails_when_capture_missing(self):
    from aTrain import cli as cli_module

    with tempfile.TemporaryDirectory() as temp_dir:
        root = Path(temp_dir)
        staging_dir = root / "staging"
        (staging_dir / "file-id").mkdir(parents=True)

        with self.assertRaisesRegex(FileNotFoundError, "Speaker embeddings were not captured"):
            cli_module._copy_speaker_embeddings(staging_dir, "file-id", root / "exported.npz")
```

Run:

```powershell
cd D:\Git\aTrain
.\.venv\Scripts\python.exe -m unittest tests.test_cli_voiceprints -v
```

Expected before implementation: `_copy_speaker_embeddings` missing or CLI option unsupported.

- [ ] **Step 2: Add helper to copy captured embeddings**

In `aTrain/cli.py`, add:

```python
def _copy_speaker_embeddings(staging_dir: Path, file_id: str, output_path: Path) -> Path:
    source = staging_dir / file_id / "_speaker_embeddings.npz"
    if not source.exists():
        raise FileNotFoundError(
            "Speaker embeddings were not captured. Use --speaker-detection and --identify-speakers."
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, output_path)
    return output_path
```

If importing `CAPTURE_FILENAME` from `voiceprint_identification.py` keeps duplication lower, use:

```python
from aTrain.voiceprint_identification import CAPTURE_FILENAME
source = staging_dir / file_id / CAPTURE_FILENAME
```

- [ ] **Step 3: Thread output path through transcription**

Add `speaker_embeddings_output: Path | None` to `_transcribe_one()` and `_run_batch()`.

In `transcribe()`, add option:

```python
speaker_embeddings_output: Annotated[
    Path | None,
    typer.Option("--speaker-embeddings-output", help="Write captured speaker embeddings to this NPZ path."),
] = None
```

Validation:

```python
if speaker_embeddings_output is not None and not speaker_detection:
    raise ValueError("--speaker-embeddings-output requires --speaker-detection.")
if speaker_embeddings_output is not None and not identify_speakers:
    raise ValueError("--speaker-embeddings-output requires --identify-speakers.")
if speaker_embeddings_output is not None and input.is_dir():
    raise ValueError("--speaker-embeddings-output currently supports single-file input only.")
```

Inside `_transcribe_one()`, call `_copy_speaker_embeddings(staging_dir, file_id, speaker_embeddings_output)` after `_identify_captured_speakers()` and before `_copy_outputs()` / cleanup. This keeps the artifact available before `shutil.rmtree(staging_dir)`.

- [ ] **Step 4: Verify export behavior**

Run:

```powershell
cd D:\Git\aTrain
.\.venv\Scripts\python.exe -m unittest tests.test_cli_voiceprints tests.test_voiceprint_identification -v
```

Expected after implementation: all selected tests pass and the export test proves the artifact survives staging cleanup.

- [ ] **Step 5: Commit export task**

Run:

```powershell
cd D:\Git\aTrain
git status -sb
git add aTrain\cli.py aTrain\voiceprint_identification.py tests\test_cli_voiceprints.py
git commit -m "Export speaker embeddings from CLI transcription"
```

If `voiceprint_identification.py` was not modified, omit it from `git add`.

---

### Task 4: Update Documentation

**Files:**
- Modify: `D:\Git\aTrain\README.md`

- [ ] **Step 1: Update README speaker voiceprint section**

Replace the stale CLI limitation text with this content adjusted to the README's existing style:

```markdown
### CLI voiceprint enrollment

The CLI can now create or update local voiceprint profiles. Profiles are stored in the same local voiceprint directory used by the GUI, and `transcribe --identify-speakers` consumes those profiles during later transcription runs.

Enroll from a direct audio sample:

```powershell
aTrain-cli voiceprint enroll --name "李想" --audio "D:\samples\li-xiang.wav" --update
```

Enroll from a captured speaker embedding exported during transcription:

```powershell
aTrain-cli transcribe "D:\input\meeting.wav" --speaker-detection --identify-speakers --speaker-embeddings-output "D:\out\meeting.speaker-embeddings.npz"
aTrain-cli voiceprint enroll --name "李想" --speaker-embeddings "D:\out\meeting.speaker-embeddings.npz" --speaker SPEAKER_01 --update
```

Low-confidence matches remain as `SPEAKER_xx`; tune `--voiceprint-threshold` and `--voiceprint-margin` when needed.
```

- [ ] **Step 2: Verify docs mention both learning sources**

Run:

```powershell
cd D:\Git\aTrain
rg --encoding utf-8 -n "voiceprint enroll|speaker-embeddings-output|SPEAKER_xx" README.md
```

Expected: all three terms are present.

- [ ] **Step 3: Commit docs**

Run:

```powershell
cd D:\Git\aTrain
git status -sb
git add README.md
git commit -m "Document CLI voiceprint enrollment"
```

Expected: commit contains only README changes.

---

### Task 5: Final Verification And Separate-Session Handoff

**Files:**
- No new source files unless previous tasks uncover a test-only fixture requirement.

- [ ] **Step 1: Run focused test suite**

Run:

```powershell
cd D:\Git\aTrain
.\.venv\Scripts\python.exe -m unittest tests.test_voiceprint_cli tests.test_voiceprints tests.test_voiceprint_identification tests.test_cli_voiceprints -v
```

Expected: all selected tests pass.

- [ ] **Step 2: Run diff hygiene**

Run:

```powershell
cd D:\Git\aTrain
git diff --check
git status -sb
```

Expected:
- `git diff --check` has no output;
- status does not include unrelated untracked directories such as `.matplotlib-cache/`, `.model-backup/`, or `dist_fixed/` in the staged set.

- [ ] **Step 3: Final implementation commit if needed**

If previous tasks were batched instead of committed task-by-task, stage only intended files:

```powershell
cd D:\Git\aTrain
git add aTrain\cli.py aTrain\voiceprint_cli.py aTrain\voiceprint_identification.py tests\test_voiceprint_cli.py tests\test_cli_voiceprints.py README.md docs\superpowers\plans\2026-05-26-atrain-cli-voiceprint-enrollment.md
git commit -m "Add CLI voiceprint enrollment"
```

- [ ] **Step 4: Hand off to collab-runtime**

After this aTrain plan is implemented and committed, return to `D:\Workspace\projects-code\collab-runtime` and execute:

```powershell
cd D:\Workspace\projects-code\collab-runtime
rtk npm --workspace recording-ingest-provider run verify -- atrain-command
```

Expected: collab-runtime can safely rely on:
- `aTrain-cli transcribe --speaker-embeddings-output`;
- `aTrain-cli voiceprint enroll --speaker-embeddings ... --speaker ... --update`;
- low-confidence identification still leaving `SPEAKER_xx` unchanged.

## Open Decisions

- No collab-runtime code should be changed in the aTrain implementation session.
- The first collab-runtime Workbench implementation should enroll from exported speaker embeddings, not from freshly clipped audio.
- Directory transcription with `--speaker-embeddings-output` is intentionally rejected in the first version because one output path cannot safely represent multiple input files. Add an output-directory mode later only when a real batch use case appears.
