import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from typer.testing import CliRunner

from aTrain.cli import cli


class CliVoiceprintTests(unittest.TestCase):
    def test_postprocess_applies_replacements_and_speaker_map_once(self):
        from aTrain.cli import _postprocess_staged_outputs

        with tempfile.TemporaryDirectory() as temp_dir:
            staging_dir = Path(temp_dir)
            transcript_dir = staging_dir / "file-id"
            transcript_dir.mkdir()
            transcript_path = transcript_dir / "transcription.json"
            transcript_path.write_text(
                json.dumps(
                    {
                        "segments": [
                            {
                                "speaker": "SPEAKER_00",
                                "text": "hello 欧喷AI",
                                "words": [
                                    {"speaker": "SPEAKER_00", "word": "欧喷AI"},
                                ],
                            }
                        ]
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

            with mock.patch("aTrain_core.outputs.create_output_files") as create_outputs:
                _postprocess_staged_outputs(
                    staging_dir=staging_dir,
                    file_id="file-id",
                    speaker_detection=True,
                    replacements={"欧喷AI": "OpenAI"},
                    speaker_map={"SPEAKER_00": "Ray"},
                )

        create_outputs.assert_called_once()
        transcript = create_outputs.call_args.args[0]
        self.assertEqual(transcript["segments"][0]["speaker"], "Ray")
        self.assertEqual(transcript["segments"][0]["text"], "hello OpenAI")
        self.assertEqual(transcript["segments"][0]["words"][0]["speaker"], "Ray")
        self.assertEqual(transcript["segments"][0]["words"][0]["word"], "OpenAI")

    def test_identify_speakers_requires_speaker_detection(self):
        runner = CliRunner()

        result = runner.invoke(
            cli,
            [
                "transcribe",
                "missing.wav",
                "--no-speaker-detection",
                "--identify-speakers",
            ],
        )

        self.assertEqual(result.exit_code, 2)
        self.assertIn("requires --speaker-detection", result.stderr)
