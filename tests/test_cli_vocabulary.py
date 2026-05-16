import json
import tempfile
import unittest
from pathlib import Path

import yaml

import aTrain.cli_vocabulary as cli_vocabulary
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

    def test_load_replacements_accepts_mapping_file(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "replace.yml"
            path.write_text(
                yaml.safe_dump({"欧喷AI": "OpenAI", "黑尔墨斯": "赫尔墨斯"}, allow_unicode=True),
                encoding="utf-8",
            )

            result = cli_vocabulary.load_replacements(path)

        self.assertEqual(result, {"欧喷AI": "OpenAI", "黑尔墨斯": "赫尔墨斯"})

    def test_load_replacements_accepts_replacements_list(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "replace.json"
            path.write_text(
                json.dumps(
                    {"replacements": [{"from": "where my tokens", "to": "WhereMyTokens"}]},
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

            result = cli_vocabulary.load_replacements(path)

        self.assertEqual(result, {"where my tokens": "WhereMyTokens"})

    def test_load_replacements_accepts_empty_json_file(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "replace.json"
            path.write_text("", encoding="utf-8")

            result = cli_vocabulary.load_replacements(path)

        self.assertEqual(result, {})

    def test_replace_map_is_empty_when_path_is_none(self):
        self.assertEqual(cli_vocabulary.load_replacements(None), {})

    def test_load_replacements_rejects_empty_source_in_mapping(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "replace.yml"
            path.write_text(yaml.safe_dump({"": "X"}, allow_unicode=True), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "empty source"):
                cli_vocabulary.load_replacements(path)

    def test_load_replacements_rejects_replacements_not_list(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "replace.json"
            path.write_text(
                json.dumps({"replacements": {"a": "b"}}, ensure_ascii=False),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "replacements.*list"):
                cli_vocabulary.load_replacements(path)

    def test_load_replacements_rejects_invalid_replacement_item(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "replace.yml"
            path.write_text(
                yaml.safe_dump({"replacements": [{"from": "a"}]}, allow_unicode=True),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "from.*to"):
                cli_vocabulary.load_replacements(path)

    def test_load_replacements_rejects_top_level_list(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "replace.yml"
            path.write_text(yaml.safe_dump(["a", "b"], allow_unicode=True), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "mapping"):
                cli_vocabulary.load_replacements(path)

    def test_load_replacements_rejects_empty_top_level_list(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "replace.yml"
            path.write_text("[]", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "mapping"):
                cli_vocabulary.load_replacements(path)

    def test_load_replacements_rejects_false_top_level_scalar(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "replace.yml"
            path.write_text("false", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "mapping"):
                cli_vocabulary.load_replacements(path)

    def test_load_replacements_rejects_zero_top_level_scalar(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "replace.yml"
            path.write_text("0", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "mapping"):
                cli_vocabulary.load_replacements(path)

    def test_load_replacements_rejects_string_top_level_scalar(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "replace.yml"
            path.write_text("bad", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "mapping"):
                cli_vocabulary.load_replacements(path)

    def test_load_replacements_rejects_non_string_mapping_values(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "replace.json"
            path.write_text(json.dumps({"a": ["bad"]}, ensure_ascii=False), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "strings"):
                cli_vocabulary.load_replacements(path)

    def test_load_replacements_rejects_non_string_replacement_item_values(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "replace.yml"
            path.write_text(
                yaml.safe_dump({"replacements": [{"from": None, "to": "x"}]}, allow_unicode=True),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "strings"):
                cli_vocabulary.load_replacements(path)

    def test_load_replacements_rejects_blank_source_in_replacements_list(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "replace.yml"
            path.write_text(
                yaml.safe_dump({"replacements": [{"from": "   ", "to": "x"}]}, allow_unicode=True),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "empty source"):
                cli_vocabulary.load_replacements(path)

    def test_load_replacements_rejects_duplicate_mapping_source(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "replace.yml"
            path.write_text("' foo ': 'A'\nfoo: 'B'\n", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "duplicate"):
                cli_vocabulary.load_replacements(path)

    def test_load_replacements_rejects_duplicate_replacements_source(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "replace.yml"
            path.write_text(
                yaml.safe_dump(
                    {"replacements": [{"from": "foo", "to": "A"}, {"from": " foo ", "to": "B"}]},
                    allow_unicode=True,
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "duplicate"):
                cli_vocabulary.load_replacements(path)

    def test_load_replacements_rejects_unknown_keys_with_replacements_list(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "replace.yml"
            path.write_text(
                yaml.safe_dump({"replacements": [], "metadata": {}}, allow_unicode=True),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "unknown|only contain"):
                cli_vocabulary.load_replacements(path)

    def test_apply_replacements_updates_segments_and_words(self):
        transcript = {
            "segments": [
                {
                    "text": "我们讨论欧喷AI和where my tokens",
                    "words": [
                        {"word": "欧喷AI"},
                        {"word": "where my tokens"},
                    ],
                }
            ]
        }

        cli_vocabulary.apply_replacements_to_transcript(
            transcript,
            {"欧喷AI": "OpenAI", "where my tokens": "WhereMyTokens"},
        )

        self.assertEqual(transcript["segments"][0]["text"], "我们讨论OpenAI和WhereMyTokens")
        self.assertEqual(transcript["segments"][0]["words"][0]["word"], "OpenAI")
        self.assertEqual(transcript["segments"][0]["words"][1]["word"], "WhereMyTokens")
