import json
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
from aTrain.voiceprints import EMBEDDING_MODEL_ID, VoiceprintProfile


class GuiVoiceprintIdentificationTests(unittest.TestCase):
    def test_transcribe_with_voiceprints_captures_and_rewrites_gui_outputs(self):
        from aTrain.utils.transcription import transcribe_with_voiceprints

        settings = types.SimpleNamespace(file_id="file-id", speaker_detection=True)
        profile = VoiceprintProfile(
            name="Ray",
            model_id=EMBEDDING_MODEL_ID,
            schema_version=1,
            embedding_dim=2,
            embedding=np.array([1.0, 0.0], dtype=np.float32),
            enrollments=[],
        )
        capture_context = CaptureContext()

        with tempfile.TemporaryDirectory() as temp_dir:
            transcript_dir = Path(temp_dir) / "file-id"
            transcript_dir.mkdir()
            (transcript_dir / "transcription.json").write_text(
                json.dumps(
                    {
                        "segments": [
                            {
                                "speaker": "SPEAKER_00",
                                "text": "hello",
                                "words": [{"speaker": "SPEAKER_00", "word": "hello"}],
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )

            with (
                mock.patch(
                    "aTrain.utils.transcription.core_outputs.TRANSCRIPT_DIR",
                    temp_dir,
                ),
                mock.patch(
                    "aTrain.utils.transcription._list_voiceprints",
                    return_value=[profile],
                ),
                mock.patch(
                    "aTrain.utils.transcription._patch_core_speaker_capture",
                    return_value=capture_context,
                ),
                mock.patch("aTrain.utils.transcription._transcribe_core") as transcribe,
                mock.patch(
                    "aTrain.utils.transcription._read_captured_embeddings",
                    return_value=(["SPEAKER_00"], np.array([[1.0, 0.0]], dtype=np.float32)),
                ),
                mock.patch(
                    "aTrain.utils.transcription._cosine_similarity_matrix",
                    return_value=np.array([[0.9]], dtype=np.float32),
                ),
                mock.patch(
                    "aTrain.utils.transcription._assign_voiceprints",
                    return_value={"SPEAKER_00": "Ray"},
                ),
                mock.patch(
                    "aTrain.utils.transcription.core_outputs.create_output_files"
                ) as create_output_files,
                mock.patch("aTrain.utils.transcription.core_outputs.write_logfile"),
            ):
                transcribe_with_voiceprints(settings)

        transcribe.assert_called_once_with(settings)
        self.assertTrue(capture_context.entered)
        self.assertTrue(capture_context.exited)
        create_output_files.assert_called_once()
        transcript = create_output_files.call_args.args[0]
        self.assertEqual(transcript["segments"][0]["speaker"], "Ray")
        self.assertEqual(transcript["segments"][0]["words"][0]["speaker"], "Ray")

    def test_transcribe_with_voiceprints_noops_when_gui_has_no_voiceprints(self):
        from aTrain.utils.transcription import transcribe_with_voiceprints

        settings = types.SimpleNamespace(file_id="file-id", speaker_detection=True)

        with (
            mock.patch(
                "aTrain.utils.transcription._list_voiceprints",
                return_value=[],
            ),
            mock.patch("aTrain.utils.transcription._patch_core_speaker_capture") as patch_capture,
            mock.patch("aTrain.utils.transcription._transcribe_core") as transcribe,
        ):
            transcribe_with_voiceprints(settings)

        transcribe.assert_called_once_with(settings)
        patch_capture.assert_not_called()


class CaptureContext:
    def __init__(self):
        self.entered = False
        self.exited = False

    def __enter__(self):
        self.entered = True
        return self

    def __exit__(self, exc_type, exc, tb):
        self.exited = True
