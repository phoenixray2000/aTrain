import tempfile
import unittest
from pathlib import Path

from aTrain.cli_vocabulary import (
    build_prompt,
    build_hotwords,
)


class VocabularyParsingTests(unittest.TestCase):
    def test_build_hotwords_deduplicates_direct_and_file_values(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            hotwords_file = Path(temp_dir) / "hotwords.txt"
            hotwords_file.write_text("OpenAI\nHermes, 飞书\nOpenAI\n", encoding="utf-8")

            result = build_hotwords("Codex, Hermes", hotwords_file)

        self.assertEqual(result, "Codex, Hermes, OpenAI, 飞书")

    def test_build_prompt_combines_direct_text_and_file_text(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            prompt_file = Path(temp_dir) / "prompt.txt"
            prompt_file.write_text("Prefer product names exactly.", encoding="utf-8")

            result = build_prompt("Meeting about transcription.", prompt_file)

        self.assertEqual(
            result,
            "Meeting about transcription.\nPrefer product names exactly.",
        )

    def test_build_prompt_accepts_file_only(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            prompt_file = Path(temp_dir) / "prompt.txt"
            prompt_file.write_text("Use the provided glossary.", encoding="utf-8")

            result = build_prompt(None, prompt_file)

        self.assertEqual(result, "Use the provided glossary.")
