import types
import unittest
from pathlib import Path
from unittest import mock

from aTrain.transcription_hotwords import (
    attach_hotwords,
    patch_core_hotwords,
    run_transcription_with_hotwords,
)
from aTrain_core.settings import Device


class HotwordsPatchTests(unittest.TestCase):
    def test_attach_hotwords_sets_dynamic_settings_attribute(self):
        settings = types.SimpleNamespace()

        attach_hotwords(settings, "OpenAI, Codex")

        self.assertEqual(settings.hotwords, "OpenAI, Codex")

    def test_patch_core_hotwords_restores_original_function(self):
        import aTrain_core.transcribe as core_transcribe

        original = core_transcribe.run_transcription
        with patch_core_hotwords("OpenAI"):
            self.assertIsNot(core_transcribe.run_transcription, original)

        self.assertIs(core_transcribe.run_transcription, original)

    def test_patch_core_hotwords_restores_original_function_after_exception(self):
        import aTrain_core.transcribe as core_transcribe

        original = core_transcribe.run_transcription
        with self.assertRaises(RuntimeError), patch_core_hotwords("OpenAI"):
            raise RuntimeError("boom")

        self.assertIs(core_transcribe.run_transcription, original)

    def test_run_transcription_passes_hotwords_to_whisper(self):
        settings = self._settings(hotwords="OpenAI, Codex")

        transcript, transcribe_mock = self._run_with_mocks(settings)

        self.assertEqual(transcript, {"segments": [{"text": "OpenAI"}]})
        self.assertEqual(transcribe_mock.call_args.kwargs["hotwords"], "OpenAI, Codex")

    def test_run_transcription_without_hotwords_passes_none(self):
        settings = self._settings()

        _, transcribe_mock = self._run_with_mocks(settings)

        self.assertIsNone(transcribe_mock.call_args.kwargs["hotwords"])

    def test_run_transcription_cpu_exception_is_reraised(self):
        settings = self._settings()

        with (
            mock.patch("aTrain.transcription_hotwords.WhisperModel") as whisper_model,
            mock.patch(
                "aTrain.transcription_hotwords.load_model_config_file",
                return_value={"base": {"type": "standard"}},
            ),
            mock.patch("aTrain.transcription_hotwords.write_logfile"),
        ):
            whisper = whisper_model.return_value
            whisper.transcribe.side_effect = RuntimeError("transcribe failed")

            with self.assertRaisesRegex(RuntimeError, "transcribe failed"):
                run_transcription_with_hotwords(settings, Path("model"), object())

    def test_run_transcription_accepts_core_return_dict_keyword_for_gpu_errors(self):
        settings = self._settings()
        settings.device = Device.GPU
        return_dict = {}

        with (
            mock.patch("aTrain.transcription_hotwords.WhisperModel") as whisper_model,
            mock.patch(
                "aTrain.transcription_hotwords.load_model_config_file",
                return_value={"base": {"type": "standard"}},
            ),
            mock.patch("aTrain.transcription_hotwords.write_logfile"),
        ):
            whisper = whisper_model.return_value
            whisper.transcribe.side_effect = RuntimeError("gpu transcribe failed")

            result = run_transcription_with_hotwords(
                settings, Path("model"), object(), returnDict=return_dict
            )

        self.assertIsNone(result)
        self.assertIsInstance(return_dict["error"], RuntimeError)
        self.assertEqual(str(return_dict["error"]), "gpu transcribe failed")

    def _settings(self, hotwords=None):
        settings = types.SimpleNamespace(
            device=Device.CPU,
            compute_type=types.SimpleNamespace(value="float32"),
            cpu_threads=1,
            model="base",
            file_id="file-id",
            language="en",
            initial_prompt=None,
            temperature=None,
            progress={},
        )
        if hotwords is not None:
            settings.hotwords = hotwords
        return settings

    def _run_with_mocks(self, settings):
        segment = {"text": "OpenAI"}
        info = types.SimpleNamespace(duration=1.0)

        with (
            mock.patch("aTrain.transcription_hotwords.WhisperModel") as whisper_model,
            mock.patch(
                "aTrain.transcription_hotwords.load_model_config_file",
                return_value={"base": {"type": "standard"}},
            ),
            mock.patch(
                "aTrain.transcription_hotwords.transcription_with_progress_bar",
                return_value=[segment],
            ),
            mock.patch(
                "aTrain.transcription_hotwords.named_tuple_to_dict",
                side_effect=lambda value: value,
            ),
            mock.patch("aTrain.transcription_hotwords.write_logfile"),
        ):
            whisper = whisper_model.return_value
            whisper.transcribe.return_value = (iter([segment]), info)

            transcript = run_transcription_with_hotwords(settings, Path("model"), object())

        return transcript, whisper.transcribe
