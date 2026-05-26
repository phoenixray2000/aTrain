import json
import shutil
import sys
import tempfile
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated

import typer
from aTrain_core.globals import (
    DEFAULT_CPU_THREADS,
    MAX_CPU_THREADS,
)
from aTrain_core.load_resources import download_all_models, get_model
from aTrain_core.settings import (
    ComputeType,
    Device,
    Settings,
    check_file,
    check_inputs_transcribe,
)
from aTrain.cli_vocabulary import (
    ReplacementMap,
    apply_replacements_to_transcript,
    build_hotwords,
    build_prompt,
    load_replacements,
)
from aTrain.model_downloads import check_model_downloaded as _check_model_downloaded
from aTrain.transcription_hotwords import attach_hotwords, patch_core_hotwords
from aTrain.voiceprint_cli import (
    enroll_voiceprint_from_audio,
    enroll_voiceprint_from_speaker_embedding,
)
from aTrain.voiceprint_identification import (
    CAPTURE_FILENAME,
    patch_core_speaker_capture,
    read_captured_embeddings,
)
from aTrain.voiceprints import (
    apply_speaker_map_to_transcript,
    assign_voiceprints,
    cosine_similarity_matrix,
    list_voiceprints,
)

cli = typer.Typer(help="CLI for aTrain.", no_args_is_help=True)
voiceprint_cli = typer.Typer(help="Manage speaker voiceprints.", no_args_is_help=True)
cli.add_typer(voiceprint_cli, name="voiceprint")

FORMAT_OUTPUTS = {
    "json": ("transcription.json", "{stem}.json"),
    "txt": ("transcription.txt", "{stem}.txt"),
    "timestamps": ("transcription_timestamps.txt", "{stem}_timestamps.txt"),
    "maxqda": ("transcription_maxqda.txt", "{stem}_maxqda.txt"),
    "srt": ("transcription.srt", "{stem}.srt"),
}
ALLOWED_FORMATS = ",".join(FORMAT_OUTPUTS)
DEFAULT_FORMATS = "txt,timestamps"


@dataclass(frozen=True)
class InputFile:
    path: Path
    display_path: Path
    relative_dir: Path


@dataclass(frozen=True)
class OutputPlan:
    format_key: str
    source_name: str
    target_path: Path


@dataclass
class FileResult:
    path: Path
    ok: bool
    reason: str = ""
    staging_dir: Path | None = None


class CliTranscriptionError(Exception):
    def __init__(self, message: str, staging_dir: Path | None = None):
        super().__init__(message)
        self.staging_dir = staging_dir


def _parse_formats(value: str) -> list[str]:
    parsed = [part.strip().lower() for part in value.split(",") if part.strip()]
    if not parsed:
        raise ValueError("No output formats specified.")
    invalid = [item for item in parsed if item not in FORMAT_OUTPUTS]
    if invalid:
        allowed = ", ".join(FORMAT_OUTPUTS)
        raise ValueError(f"Unsupported output format(s): {', '.join(invalid)}. Allowed: {allowed}")
    return list(dict.fromkeys(parsed))


def _is_supported_file(path: Path) -> bool:
    return path.is_file() and check_file(path.name)


def _collect_inputs(input_path: Path, recursive: bool) -> tuple[list[InputFile], list[Path]]:
    if not input_path.exists():
        raise ValueError(f"Input does not exist: {input_path}")

    if input_path.is_file():
        if not _is_supported_file(input_path):
            raise ValueError(f"Input file extension is not supported: {input_path}")
        return [InputFile(input_path, Path(input_path.name), Path("."))], []

    if not input_path.is_dir():
        raise ValueError(f"Input is neither a file nor a directory: {input_path}")

    root = input_path.resolve()
    candidates = root.rglob("*") if recursive else root.iterdir()
    files = [path for path in candidates if path.is_file()]
    inputs: list[InputFile] = []
    skipped: list[Path] = []
    for path in sorted(files, key=lambda item: str(item).lower()):
        if not _is_supported_file(path):
            skipped.append(path)
            continue
        relative_dir = path.parent.resolve().relative_to(root) if recursive else Path(".")
        display_path = relative_dir / path.name if relative_dir != Path(".") else Path(path.name)
        inputs.append(InputFile(path, display_path, relative_dir))

    if not inputs:
        raise ValueError(f"No supported audio/video files found in: {input_path}")

    return inputs, skipped


def _resolve_output_dirs(
    output: Path,
    json_output: Path | None,
    txt_output: Path | None,
    timestamps_output: Path | None,
    maxqda_output: Path | None,
    srt_output: Path | None,
) -> dict[str, Path]:
    return {
        "json": json_output or output,
        "txt": txt_output or output,
        "timestamps": timestamps_output or output,
        "maxqda": maxqda_output or output,
        "srt": srt_output or output,
    }


def _build_output_plan(
    item: InputFile, formats: list[str], output_dirs: dict[str, Path]
) -> list[OutputPlan]:
    plans = []
    for format_key in formats:
        source_name, target_template = FORMAT_OUTPUTS[format_key]
        target_dir = output_dirs[format_key] / item.relative_dir
        target_name = target_template.format(stem=item.path.stem)
        plans.append(OutputPlan(format_key, source_name, target_dir / target_name))
    return plans


def _copy_outputs(
    staging_dir: Path,
    file_id: str,
    output_plan: list[OutputPlan],
    overwrite: bool,
) -> None:
    source_dir = staging_dir / file_id
    for planned in output_plan:
        source_path = source_dir / planned.source_name
        if not source_path.exists():
            raise FileNotFoundError(
                f"Expected {planned.format_key} output was not created: {source_path}"
            )
        if planned.target_path.exists() and not overwrite:
            raise FileExistsError(
                f"Target file exists: {planned.target_path}. Use --overwrite to replace it."
            )

    for planned in output_plan:
        planned.target_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_dir / planned.source_name, planned.target_path)


def _copy_speaker_embeddings(staging_dir: Path, file_id: str, output_path: Path) -> Path:
    source = staging_dir / file_id / CAPTURE_FILENAME
    if not source.exists():
        raise FileNotFoundError(
            "Speaker embeddings were not captured. Use --speaker-detection and --identify-speakers."
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, output_path)
    return output_path


def _postprocess_staged_outputs(
    staging_dir: Path,
    file_id: str,
    replacements: ReplacementMap,
    speaker_detection: bool,
    speaker_map: dict[str, str] | None,
) -> None:
    if not replacements and not speaker_map:
        return

    from aTrain_core import outputs as core_outputs

    transcript_path = staging_dir / file_id / "transcription.json"
    with transcript_path.open("r", encoding="utf-8") as handle:
        transcript = json.load(handle)
    apply_replacements_to_transcript(transcript, replacements)
    if speaker_map:
        apply_speaker_map_to_transcript(transcript, speaker_map)
    core_outputs.create_output_files(transcript, speaker_detection, file_id)


def _transcribe_one(
    item: InputFile,
    output_plan: list[OutputPlan],
    overwrite: bool,
    model: str,
    language: str,
    speaker_detection: bool,
    speaker_count: int,
    device: Device,
    compute_type: ComputeType,
    temperature: float | None,
    prompt: str | None,
    hotwords: str | None,
    replacements: ReplacementMap,
    identify_speakers: bool,
    voiceprint_threshold: float,
    voiceprint_margin: float,
    speaker_embeddings_output: Path | None,
    cpu_threads: int,
) -> Path:
    for planned in output_plan:
        if planned.target_path.exists() and not overwrite:
            raise FileExistsError(
                f"Target file exists: {planned.target_path}. Use --overwrite to replace it."
            )

    from aTrain_core import outputs as core_outputs
    from aTrain_core.transcribe import prepare_transcription, transcribe as transcribe_core

    staging_dir = Path(tempfile.mkdtemp(prefix="atrain-cli-"))
    original_transcript_dir = core_outputs.TRANSCRIPT_DIR
    gpu_log_dir: Path | None = None
    gpu_log_dir_created = False
    core_outputs.TRANSCRIPT_DIR = str(staging_dir)
    try:
        _, file_id, timestamp = prepare_transcription(item.path)
        # prepare_transcription sanitizes names for file_id; decoding must use the real path.
        file = item.path
        if device == Device.GPU:
            gpu_log_dir = Path(original_transcript_dir) / file_id
            if not gpu_log_dir.exists():
                gpu_log_dir.mkdir(parents=True, exist_ok=True)
                gpu_log_dir_created = True
        check_inputs_transcribe(str(file), model, language, device)
        settings = Settings(
            file=file,
            file_id=file_id,
            file_name=file.name,
            model=model,
            language=language,
            speaker_detection=speaker_detection,
            speaker_count=speaker_count or None,
            device=device,
            compute_type=compute_type,
            timestamp=timestamp,
            temperature=temperature,
            initial_prompt=prompt,
            progress={},
            cpu_threads=cpu_threads,
        )
        attach_hotwords(settings, hotwords)
        capture_speakers = identify_speakers and speaker_detection
        with patch_core_hotwords(hotwords):
            if capture_speakers:
                with patch_core_speaker_capture():
                    transcribe_core(settings)
            else:
                transcribe_core(settings)

        speaker_map = _identify_captured_speakers(
            staging_dir=staging_dir,
            file_id=file_id,
            enabled=capture_speakers,
            threshold=voiceprint_threshold,
            margin=voiceprint_margin,
        )
        if speaker_embeddings_output is not None:
            _copy_speaker_embeddings(staging_dir, file_id, speaker_embeddings_output)
        _postprocess_staged_outputs(
            staging_dir,
            file_id,
            replacements,
            speaker_detection,
            speaker_map,
        )
        _copy_outputs(staging_dir, file_id, output_plan, overwrite)
        shutil.rmtree(staging_dir, ignore_errors=True)
        return staging_dir
    except Exception as error:
        raise CliTranscriptionError(str(error), staging_dir) from error
    finally:
        core_outputs.TRANSCRIPT_DIR = original_transcript_dir
        if gpu_log_dir_created and gpu_log_dir is not None:
            shutil.rmtree(gpu_log_dir, ignore_errors=True)


def _identify_captured_speakers(
    staging_dir: Path,
    file_id: str,
    enabled: bool,
    threshold: float,
    margin: float,
) -> dict[str, str] | None:
    if not enabled:
        return None
    captured = read_captured_embeddings(staging_dir, file_id)
    if captured is None:
        return None

    labels, embeddings = captured
    voiceprints = list_voiceprints()
    if not labels or not voiceprints:
        return None

    scores = cosine_similarity_matrix(embeddings, voiceprints)
    speaker_map = assign_voiceprints(
        scores,
        labels,
        [profile.name for profile in voiceprints],
        threshold,
        margin,
    )
    return speaker_map or None


def _run_batch(
    inputs: list[InputFile],
    skipped: list[Path],
    formats: list[str],
    output_dirs: dict[str, Path],
    overwrite: bool,
    model: str,
    language: str,
    speaker_detection: bool,
    speaker_count: int,
    device: Device,
    compute_type: ComputeType,
    temperature: float | None,
    prompt: str | None,
    hotwords: str | None,
    replacements: ReplacementMap,
    identify_speakers: bool,
    voiceprint_threshold: float,
    voiceprint_margin: float,
    speaker_embeddings_output: Path | None,
    cpu_threads: int,
) -> int:
    results: list[FileResult] = []
    total = len(inputs)

    for index, item in enumerate(inputs, 1):
        output_plan = _build_output_plan(item, formats, output_dirs)
        typer.echo(f"[{index}/{total}] start: {item.display_path}")
        started = time.monotonic()
        staging_dir: Path | None = None
        try:
            staging_dir = _transcribe_one(
                item=item,
                output_plan=output_plan,
                overwrite=overwrite,
                model=model,
                language=language,
                speaker_detection=speaker_detection,
                speaker_count=speaker_count,
                device=device,
                compute_type=compute_type,
                temperature=temperature,
                prompt=prompt,
                hotwords=hotwords,
                replacements=replacements,
                identify_speakers=identify_speakers,
                voiceprint_threshold=voiceprint_threshold,
                voiceprint_margin=voiceprint_margin,
                speaker_embeddings_output=speaker_embeddings_output,
                cpu_threads=cpu_threads,
            )
            elapsed = int(time.monotonic() - started)
            typer.echo(f"[{index}/{total}] done: {item.display_path} ({elapsed}s)")
            results.append(FileResult(item.path, ok=True))
        except Exception as error:
            elapsed = int(time.monotonic() - started)
            reason = str(error)
            staging_dir = getattr(error, "staging_dir", staging_dir)
            typer.echo(
                f"[{index}/{total}] FAIL: {item.display_path} ({elapsed}s) - {reason}",
                err=True,
            )
            traceback.print_exc(file=sys.stderr)
            results.append(FileResult(item.path, ok=False, reason=reason, staging_dir=staging_dir))

    succeeded = sum(1 for result in results if result.ok)
    failed = len(results) - succeeded
    _print_summary(results, skipped)

    if failed and succeeded:
        return 1
    if failed or not succeeded:
        return 2
    return 0


def _print_summary(results: list[FileResult], skipped: list[Path]) -> None:
    succeeded = sum(1 for result in results if result.ok)
    failed = len(results) - succeeded
    typer.echo("\nSummary:")
    typer.echo(f"  total:     {len(results)}")
    typer.echo(f"  succeeded: {succeeded}")
    typer.echo(f"  failed:    {failed}")
    typer.echo(f"  skipped:   {len(skipped)}")

    if skipped:
        typer.echo("Skipped:")
        for path in skipped:
            typer.echo(f"  - {path} - extension not supported")

    failures = [result for result in results if not result.ok]
    if failures:
        typer.echo("Failures:", err=True)
        for result in failures:
            staging = f"; staging={result.staging_dir}" if result.staging_dir else ""
            typer.echo(f"  - {result.path} - {result.reason}{staging}", err=True)


@voiceprint_cli.command("enroll")
def voiceprint_enroll(
    name: Annotated[str, typer.Option("--name", help="Person name for the voiceprint.")],
    audio: Annotated[
        Path | None,
        typer.Option("--audio", help="Audio sample used for enrollment."),
    ] = None,
    speaker_embeddings: Annotated[
        Path | None,
        typer.Option("--speaker-embeddings", help="NPZ speaker embedding artifact exported by transcribe."),
    ] = None,
    speaker: Annotated[
        str | None,
        typer.Option("--speaker", help="Speaker label to enroll, such as SPEAKER_01."),
    ] = None,
    update: Annotated[
        bool,
        typer.Option("--update", help="Merge into an existing profile."),
    ] = False,
    source: Annotated[
        str | None,
        typer.Option("--source", help="Optional audit source stored with the enrollment."),
    ] = None,
    device: Annotated[Device, typer.Option(help="Hardware used for audio embedding extraction.")] = Device.CPU,
    min_duration_sec: Annotated[
        float,
        typer.Option("--min-duration-sec", help="Minimum audio duration accepted for direct audio enrollment.", min=0.1),
    ] = 3.0,
):
    """Create or update a local speaker voiceprint."""
    try:
        source_count = int(audio is not None) + int(speaker_embeddings is not None)
        if source_count != 1:
            raise ValueError("Provide exactly one of --audio or --speaker-embeddings.")
        if speaker_embeddings is not None and not speaker:
            raise ValueError("--speaker is required with --speaker-embeddings.")
        if audio is not None and speaker:
            raise ValueError("--speaker can only be used with --speaker-embeddings.")
        if audio is not None:
            profile = enroll_voiceprint_from_audio(
                audio,
                name,
                update=update,
                device=device,
                min_duration_sec=min_duration_sec,
            )
        else:
            profile = enroll_voiceprint_from_speaker_embedding(
                speaker_embeddings,
                speaker,
                name,
                update=update,
                source=source,
            )
    except Exception as error:
        typer.echo(str(error), err=True)
        raise typer.Exit(code=2) from error

    typer.echo(f"Voiceprint enrolled: {getattr(profile, 'name', name)}")


@cli.command()
def transcribe(
    input: Annotated[Path, typer.Argument(help="Audio/video file or directory to transcribe.")],
    model: Annotated[str, typer.Option(help="Whisper model used to transcribe.")] = "large-v3",
    language: Annotated[str, typer.Option(help="Language of the audio.")] = "auto-detect",
    prompt: Annotated[str | None, typer.Option(help="Initial prompt passed to model.")] = None,
    prompt_file: Annotated[
        Path | None,
        typer.Option("--prompt-file", help="UTF-8 text file appended to --prompt."),
    ] = None,
    hotwords: Annotated[
        str | None,
        typer.Option("--hotwords", help="Comma- or newline-separated hot words passed to faster-whisper."),
    ] = None,
    hotwords_file: Annotated[
        Path | None,
        typer.Option("--hotwords-file", help="UTF-8 file containing comma- or newline-separated hot words."),
    ] = None,
    replace_map: Annotated[
        Path | None,
        typer.Option("--replace-map", help="JSON/YAML replacement map applied after transcription."),
    ] = None,
    speaker_detection: Annotated[
        bool,
        typer.Option(
            "--speaker-detection/--no-speaker-detection",
            help="Enable speaker detection.",
        ),
    ] = True,
    speaker_count: Annotated[
        int,
        typer.Option(help="Number of speakers. Use 0 to let aTrain auto-detect."),
    ] = 0,
    identify_speakers: Annotated[
        bool,
        typer.Option(
            "--identify-speakers/--no-identify-speakers",
            help="Rename diarized SPEAKER_xx labels using enrolled voiceprints.",
        ),
    ] = True,
    voiceprint_threshold: Annotated[
        float,
        typer.Option(
            "--voiceprint-threshold",
            help="Minimum cosine similarity required for voiceprint identification.",
            min=0.0,
            max=1.0,
        ),
    ] = 0.5,
    voiceprint_margin: Annotated[
        float,
        typer.Option(
            "--voiceprint-margin",
            help="Minimum score gap over competing speaker/name assignments.",
            min=0.0,
            max=1.0,
        ),
    ] = 0.05,
    speaker_embeddings_output: Annotated[
        Path | None,
        typer.Option("--speaker-embeddings-output", help="Write captured speaker embeddings to this NPZ path."),
    ] = None,
    device: Annotated[Device, typer.Option(help="Hardware used to transcribe.")] = Device.GPU,
    compute_type: Annotated[
        ComputeType, typer.Option(help="Data type used in computations.")
    ] = ComputeType.FLOAT32,
    temperature: Annotated[
        float | None, typer.Option(help="Temperature used for sampling.", min=0.0, max=1.0)
    ] = None,
    cpu_threads: Annotated[
        int,
        typer.Option(
            help=f"Number of CPU threads to use (0 = auto, default is {DEFAULT_CPU_THREADS}).",
            min=0,
            max=MAX_CPU_THREADS,
        ),
    ] = DEFAULT_CPU_THREADS,
    recursive: Annotated[
        bool, typer.Option(help="Recursively scan directories for supported files.")
    ] = False,
    formats: Annotated[
        str,
        typer.Option(help=f"Comma-separated output formats. Allowed: {ALLOWED_FORMATS}."),
    ] = DEFAULT_FORMATS,
    output: Annotated[Path, typer.Option(help="Default output directory.")] = Path(
        "atrain-output"
    ),
    json_output: Annotated[Path | None, typer.Option(help="JSON output directory.")] = None,
    txt_output: Annotated[Path | None, typer.Option(help="Plain TXT output directory.")] = None,
    timestamps_output: Annotated[
        Path | None, typer.Option(help="Timestamped TXT output directory.")
    ] = None,
    maxqda_output: Annotated[Path | None, typer.Option(help="MAXQDA TXT output directory.")] = None,
    srt_output: Annotated[Path | None, typer.Option(help="SRT output directory.")] = None,
    overwrite: Annotated[bool, typer.Option(help="Overwrite existing output files.")] = True,
):
    """Transcribe a single file or a directory of files."""
    try:
        selected_formats = _parse_formats(formats)
        prompt_value = build_prompt(prompt, prompt_file)
        hotwords_value = build_hotwords(hotwords, hotwords_file)
        replacements = load_replacements(replace_map)
        if identify_speakers and not speaker_detection:
            raise ValueError("--identify-speakers requires --speaker-detection.")
        if speaker_embeddings_output is not None and not speaker_detection:
            raise ValueError("--speaker-embeddings-output requires --speaker-detection.")
        if speaker_embeddings_output is not None and not identify_speakers:
            raise ValueError("--speaker-embeddings-output requires --identify-speakers.")
        if speaker_embeddings_output is not None and input.is_dir():
            raise ValueError("--speaker-embeddings-output currently supports single-file input only.")
        inputs, skipped = _collect_inputs(input, recursive)
        _check_model_downloaded(model)
        if speaker_detection:
            _check_model_downloaded("speaker-detection")
        output_dirs = _resolve_output_dirs(
            output=output,
            json_output=json_output,
            txt_output=txt_output,
            timestamps_output=timestamps_output,
            maxqda_output=maxqda_output,
            srt_output=srt_output,
        )
    except Exception as error:
        typer.echo(str(error), err=True)
        raise typer.Exit(code=2) from error

    exit_code = _run_batch(
        inputs=inputs,
        skipped=skipped,
        formats=selected_formats,
        output_dirs=output_dirs,
        overwrite=overwrite,
        model=model,
        language=language,
        speaker_detection=speaker_detection,
        speaker_count=speaker_count,
        device=device,
        compute_type=compute_type,
        temperature=temperature,
        prompt=prompt_value,
        hotwords=hotwords_value,
        replacements=replacements,
        identify_speakers=identify_speakers,
        voiceprint_threshold=voiceprint_threshold,
        voiceprint_margin=voiceprint_margin,
        speaker_embeddings_output=speaker_embeddings_output,
        cpu_threads=cpu_threads,
    )
    raise typer.Exit(code=exit_code)


@cli.command()
def init(
    model: Annotated[
        str, typer.Argument(help="Model to download, or 'all'.")
    ] = "large-v3-turbo",
):
    """Download a model for CLI/GUI use."""
    try:
        if model == "all":
            download_all_models()
            typer.echo("All models downloaded")
        else:
            get_model(model)
            typer.echo(f"Model {model} downloaded")
    except Exception as error:
        typer.echo(str(error), err=True)
        raise typer.Exit(code=2) from error


if __name__ == "__main__":
    cli()
