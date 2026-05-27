import tempfile
import unittest
from pathlib import Path
from unittest import mock

from aTrain.cli import InputFile, _transcribe_one
from aTrain_core.settings import ComputeType, Device


class CliPathTests(unittest.TestCase):
    def test_transcribe_one_preserves_original_file_path_with_unicode_spaces(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            audio_path = Path(temp_dir) / "录音 (28).m4a"
            audio_path.write_bytes(b"placeholder")
            item = InputFile(audio_path, Path(audio_path.name), Path("."))

            def assert_original_path(path, *args):
                self.assertEqual(str(audio_path), path)

            def assert_original_settings(settings):
                self.assertEqual(audio_path, settings.file)
                self.assertEqual(audio_path.name, settings.file_name)

            with (
                mock.patch("aTrain.cli.check_inputs_transcribe", side_effect=assert_original_path),
                mock.patch("aTrain_core.transcribe.transcribe", side_effect=assert_original_settings),
            ):
                _transcribe_one(
                    item=item,
                    output_plan=[],
                    overwrite=True,
                    model="large-v3-turbo",
                    language="auto-detect",
                    speaker_detection=False,
                    speaker_count=0,
                    device=Device.CPU,
                    compute_type=ComputeType.FLOAT32,
                    temperature=None,
                    prompt=None,
                    cpu_threads=0,
                )
