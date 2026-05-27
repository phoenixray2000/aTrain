import os
from collections.abc import Iterator
from contextlib import contextmanager
from multiprocessing.managers import DictProxy
from pathlib import Path

import numpy as np
from aTrain_core.load_resources import load_model_config_file
from aTrain_core.outputs import named_tuple_to_dict, write_logfile
from aTrain_core.settings import Device, Settings
from aTrain_core.transcribe import transcription_with_progress_bar
from faster_whisper import WhisperModel


def attach_hotwords(settings: Settings, hotwords: str | None) -> None:
    if hotwords:
        settings.hotwords = hotwords


def run_transcription_with_hotwords(
    settings: Settings,
    model_path: Path,
    audio_array: np.ndarray,
    returnDict: DictProxy | dict | None = None,  # noqa: N803 - matches aTrain_core kwarg.
) -> dict | None:
    """Run a transcription using a whisper model with optional hotwords."""
    return_dict = {} if returnDict is None else returnDict

    try:
        whisper_model = WhisperModel(
            model_size_or_path=model_path.as_posix(),
            device="cuda" if settings.device == Device.GPU else "cpu",
            compute_type=settings.compute_type.value,
            cpu_threads=settings.cpu_threads,
        )
        model_type = load_model_config_file()[settings.model]["type"]
        write_logfile(f"Transcribing with {model_type} model.", settings.file_id)

        segments, info = whisper_model.transcribe(
            audio=audio_array,
            vad_filter=True,
            beam_size=5,
            word_timestamps=True,
            language=None if settings.language == "auto-detect" else settings.language,
            max_new_tokens=None if model_type == "distil" else 128,
            no_speech_threshold=0.6,
            condition_on_previous_text=model_type != "distil",
            initial_prompt=settings.initial_prompt,
            temperature=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
            if settings.temperature is None
            else settings.temperature,
            hotwords=getattr(settings, "hotwords", None),
        )
        segments = transcription_with_progress_bar(segments, info, settings.progress)
        transcript = {"segments": [named_tuple_to_dict(s) for s in segments]}
        write_logfile("Transcription successful", settings.file_id)
        if settings.device == Device.CPU:
            return transcript
        if settings.device == Device.GPU:
            return_dict["transcript"] = transcript
            os._exit(0)

    except Exception as error:
        if settings.device == Device.CPU:
            raise error
        if settings.device == Device.GPU:
            return_dict["error"] = error
            return None


@contextmanager
def patch_core_hotwords(hotwords: str | None) -> Iterator[None]:
    if not hotwords:
        yield
        return

    import aTrain_core.transcribe as core_transcribe

    original = core_transcribe.run_transcription
    core_transcribe.run_transcription = run_transcription_with_hotwords
    try:
        yield
    finally:
        core_transcribe.run_transcription = original
