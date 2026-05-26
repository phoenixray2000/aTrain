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
