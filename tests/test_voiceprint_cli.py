import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from aTrain.voiceprints import (
    EMBEDDING_MODEL_ID,
    VOICEPRINT_SCHEMA_VERSION,
    VoiceprintProfile,
)
from aTrain_core.settings import Device


class VoiceprintCliTests(unittest.TestCase):
    def test_audio_enrollment_creates_new_profile(self):
        from aTrain.voiceprint_cli import enroll_voiceprint_from_audio

        vector = np.array([1.0, 0.0], dtype=np.float32)
        with (
            patch("aTrain.voiceprint_cli.check_model_downloaded"),
            patch("aTrain.voiceprint_cli.get_model", return_value=Path("speaker-detection")),
            patch("aTrain.voiceprint_cli.extract_embedding", return_value=vector),
            patch(
                "aTrain.voiceprint_cli.load_voiceprint", side_effect=FileNotFoundError("missing")
            ),
            patch("aTrain.voiceprint_cli.save_voiceprint") as save,
        ):
            profile = enroll_voiceprint_from_audio(
                Path("sample.wav"), "李想", update=False, device="CPU"
            )

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
        with (
            patch("aTrain.voiceprint_cli.check_model_downloaded"),
            patch("aTrain.voiceprint_cli.get_model", return_value=Path("speaker-detection")),
            patch(
                "aTrain.voiceprint_cli.extract_embedding",
                return_value=np.array([0.0, 1.0], dtype=np.float32),
            ),
            patch("aTrain.voiceprint_cli.load_voiceprint", return_value=existing),
            self.assertRaises(FileExistsError),
        ):
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
        with (
            patch("aTrain.voiceprint_cli.check_model_downloaded"),
            patch("aTrain.voiceprint_cli.get_model", return_value=Path("speaker-detection")),
            patch(
                "aTrain.voiceprint_cli.extract_embedding",
                return_value=np.array([0.0, 1.0], dtype=np.float32),
            ),
            patch("aTrain.voiceprint_cli.load_voiceprint", return_value=existing),
            patch("aTrain.voiceprint_cli.save_voiceprint") as save,
        ):
            profile = enroll_voiceprint_from_audio(
                Path("sample.wav"), "李想", update=True, device="CPU"
            )

        np.testing.assert_allclose(profile.embedding, np.array([0.5, 0.5], dtype=np.float32))
        self.assertEqual(len(profile.enrollments), 2)
        save.assert_called_once()

    def test_speaker_embedding_enrollment_selects_requested_label(self):
        from aTrain.voiceprint_cli import enroll_voiceprint_from_speaker_embedding

        labels = ["SPEAKER_00", "SPEAKER_01"]
        embeddings = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        with (
            patch("aTrain.voiceprint_cli.read_embedding_file", return_value=(labels, embeddings)),
            patch(
                "aTrain.voiceprint_cli.load_voiceprint", side_effect=FileNotFoundError("missing")
            ),
            patch("aTrain.voiceprint_cli.save_voiceprint"),
        ):
            profile = enroll_voiceprint_from_speaker_embedding(
                Path("speakers.npz"), "SPEAKER_01", "李想", update=False
            )

        np.testing.assert_allclose(profile.embedding, np.array([0.0, 1.0], dtype=np.float32))
        self.assertEqual(profile.enrollments[0]["speaker_label"], "SPEAKER_01")

    def test_speaker_embedding_enrollment_rejects_missing_label(self):
        from aTrain.voiceprint_cli import enroll_voiceprint_from_speaker_embedding

        with (
            patch(
                "aTrain.voiceprint_cli.read_embedding_file",
                return_value=(["SPEAKER_00"], np.array([[1.0, 0.0]], dtype=np.float32)),
            ),
            self.assertRaisesRegex(ValueError, "Speaker label not found"),
        ):
            enroll_voiceprint_from_speaker_embedding(
                Path("speakers.npz"), "SPEAKER_01", "李想", update=False
            )

    def test_audio_enrollment_checks_model_before_resolving_path(self):
        from aTrain.voiceprint_cli import enroll_voiceprint_from_audio

        with (
            patch(
                "aTrain.voiceprint_cli.check_model_downloaded",
                side_effect=FileNotFoundError("missing"),
            ) as check,
            patch("aTrain.voiceprint_cli.get_model") as get_model,
            patch("aTrain.voiceprint_cli.extract_embedding") as extract,
            self.assertRaisesRegex(FileNotFoundError, "missing"),
        ):
            enroll_voiceprint_from_audio(
                Path("sample.wav"), "李想", update=False, device=Device.CPU
            )

        check.assert_called_once_with("speaker-detection")
        get_model.assert_not_called()
        extract.assert_not_called()

    def test_audio_enrollment_maps_cli_cpu_device_to_torch_device(self):
        from aTrain.voiceprint_cli import enroll_voiceprint_from_audio

        with (
            patch("aTrain.voiceprint_cli.check_model_downloaded"),
            patch("aTrain.voiceprint_cli.get_model", return_value=Path("speaker-detection")),
            patch(
                "aTrain.voiceprint_cli.extract_embedding",
                return_value=np.array([1.0, 0.0], dtype=np.float32),
            ) as extract,
            patch(
                "aTrain.voiceprint_cli.load_voiceprint", side_effect=FileNotFoundError("missing")
            ),
            patch("aTrain.voiceprint_cli.save_voiceprint"),
        ):
            enroll_voiceprint_from_audio(
                Path("sample.wav"), "李想", update=False, device=Device.CPU
            )

        self.assertEqual(extract.call_args.args[2], torch.device("cpu"))

    def test_audio_enrollment_maps_cli_gpu_device_to_cuda_when_available(self):
        from aTrain.voiceprint_cli import enroll_voiceprint_from_audio

        with (
            patch("aTrain.voiceprint_cli.check_model_downloaded"),
            patch("aTrain.voiceprint_cli.torch.cuda.is_available", return_value=True),
            patch("aTrain.voiceprint_cli.get_model", return_value=Path("speaker-detection")),
            patch(
                "aTrain.voiceprint_cli.extract_embedding",
                return_value=np.array([1.0, 0.0], dtype=np.float32),
            ) as extract,
            patch(
                "aTrain.voiceprint_cli.load_voiceprint", side_effect=FileNotFoundError("missing")
            ),
            patch("aTrain.voiceprint_cli.save_voiceprint"),
        ):
            enroll_voiceprint_from_audio(
                Path("sample.wav"), "李想", update=False, device=Device.GPU
            )

        self.assertEqual(extract.call_args.args[2], torch.device("cuda"))

    def test_audio_enrollment_rejects_unavailable_gpu_before_extracting(self):
        from aTrain.voiceprint_cli import enroll_voiceprint_from_audio

        with (
            patch("aTrain.voiceprint_cli.check_model_downloaded"),
            patch("aTrain.voiceprint_cli.torch.cuda.is_available", return_value=False),
            patch("aTrain.voiceprint_cli.get_model", return_value=Path("speaker-detection")),
            patch("aTrain.voiceprint_cli.extract_embedding") as extract,
            self.assertRaisesRegex(ValueError, "GPU is not available"),
        ):
            enroll_voiceprint_from_audio(
                Path("sample.wav"), "李想", update=False, device=Device.GPU
            )

        extract.assert_not_called()


if __name__ == "__main__":
    unittest.main()
