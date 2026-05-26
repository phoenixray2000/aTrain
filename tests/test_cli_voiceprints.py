import tempfile
import unittest
from pathlib import Path
from unittest import mock

from typer.testing import CliRunner

from aTrain.cli import cli


class CliVoiceprintTests(unittest.TestCase):
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
