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

            with mock.patch(
                "aTrain_core.outputs.create_output_files"
            ) as create_outputs:
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

    def test_speaker_embeddings_output_requires_speaker_detection(self):
        runner = CliRunner()

        result = runner.invoke(
            cli,
            [
                "transcribe",
                "missing.wav",
                "--speaker-embeddings-output",
                "out.npz",
                "--no-speaker-detection",
            ],
        )

        self.assertEqual(result.exit_code, 2)
        self.assertIn(
            "--speaker-embeddings-output requires --speaker-detection",
            result.stderr,
        )

    def test_speaker_embeddings_output_requires_identify_speakers(self):
        runner = CliRunner()

        result = runner.invoke(
            cli,
            [
                "transcribe",
                "missing.wav",
                "--speaker-embeddings-output",
                "out.npz",
                "--no-identify-speakers",
            ],
        )

        self.assertEqual(result.exit_code, 2)
        self.assertIn(
            "--speaker-embeddings-output requires --identify-speakers",
            result.stderr,
        )

    def test_speaker_embeddings_output_rejects_directory_input(self):
        runner = CliRunner()

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            output_path = root / "out.npz"

            result = runner.invoke(
                cli,
                [
                    "transcribe",
                    str(root),
                    "--speaker-embeddings-output",
                    str(output_path),
                ],
            )

        self.assertEqual(result.exit_code, 2)
        self.assertIn("single-file input only", result.stderr)

    def test_speaker_embeddings_output_passes_path_to_run_batch(self):
        runner = CliRunner()

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            audio_path = root / "sample.wav"
            audio_path.write_bytes(b"wav-stub")
            output_path = root / "speaker-embeddings.npz"

            with (
                mock.patch("aTrain.cli._check_model_downloaded"),
                mock.patch("aTrain.cli._run_batch", return_value=0) as run_batch,
            ):
                result = runner.invoke(
                    cli,
                    [
                        "transcribe",
                        str(audio_path),
                        "--speaker-embeddings-output",
                        str(output_path),
                    ],
                )

        self.assertEqual(result.exit_code, 0)
        self.assertEqual(
            run_batch.call_args.kwargs["speaker_embeddings_output"], output_path
        )
        self.assertIs(run_batch.call_args.kwargs["speaker_detection"], True)
        self.assertIs(run_batch.call_args.kwargs["identify_speakers"], True)

    def test_voiceprint_enroll_rejects_missing_source(self):
        runner = CliRunner()

        result = runner.invoke(cli, ["voiceprint", "enroll", "--name", "李想"])

        self.assertEqual(result.exit_code, 2)
        self.assertIn(
            "Provide exactly one of --audio or --speaker-embeddings", result.stderr
        )

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
        self.assertIn(
            "Provide exactly one of --audio or --speaker-embeddings", result.stderr
        )

    def test_voiceprint_enroll_from_speaker_embedding_calls_helper(self):
        runner = CliRunner()
        called = {}

        def fake_enroll(path, speaker, name, update, source=None):
            called.update(
                path=path, speaker=speaker, name=name, update=update, source=source
            )

        with tempfile.TemporaryDirectory() as temp_dir:
            embedding_file = Path(temp_dir) / "speakers.npz"
            embedding_file.write_bytes(b"npz-stub")
            with mock.patch(
                "aTrain.cli.enroll_voiceprint_from_speaker_embedding",
                side_effect=fake_enroll,
            ):
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

            result = cli_module._copy_speaker_embeddings(
                staging_dir, "file-id", output_path
            )

            self.assertEqual(output_path.read_bytes(), b"npz-data")
            self.assertEqual(result, output_path)

    def test_copy_speaker_embeddings_fails_when_capture_missing(self):
        from aTrain import cli as cli_module

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            staging_dir = root / "staging"
            (staging_dir / "file-id").mkdir(parents=True)

            with self.assertRaisesRegex(
                FileNotFoundError, "Speaker embeddings were not captured"
            ):
                cli_module._copy_speaker_embeddings(
                    staging_dir, "file-id", root / "exported.npz"
                )

    def test_copy_speaker_embeddings_respects_no_overwrite(self):
        from aTrain import cli as cli_module

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            staging_dir = root / "staging"
            file_dir = staging_dir / "file-id"
            file_dir.mkdir(parents=True)
            (file_dir / "_speaker_embeddings.npz").write_bytes(b"new-data")
            output_path = root / "exported.npz"
            output_path.write_bytes(b"old-data")

            with self.assertRaisesRegex(FileExistsError, "Target file exists"):
                cli_module._copy_speaker_embeddings(
                    staging_dir, "file-id", output_path, overwrite=False
                )

            self.assertEqual(output_path.read_bytes(), b"old-data")
