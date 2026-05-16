import json
from pathlib import Path
from typing import Any

import yaml


ReplacementMap = dict[str, str]


def read_text_file(path: Path) -> str:
    return path.read_text(encoding="utf-8").strip()


def split_terms(value: str | None) -> list[str]:
    if not value:
        return []
    normalized = value.replace("\r\n", "\n").replace("\r", "\n")
    parts: list[str] = []
    for line in normalized.split("\n"):
        parts.extend(item.strip() for item in line.split(","))
    return [item for item in parts if item]


def dedupe_preserve_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        key = value.casefold()
        if key in seen:
            continue
        seen.add(key)
        result.append(value)
    return result


def build_hotwords(hotwords: str | None, hotwords_file: Path | None) -> str | None:
    terms = split_terms(hotwords)
    if hotwords_file is not None:
        terms.extend(split_terms(read_text_file(hotwords_file)))
    terms = dedupe_preserve_order(terms)
    return ", ".join(terms) if terms else None


def build_prompt(prompt: str | None, prompt_file: Path | None) -> str | None:
    parts: list[str] = []
    if prompt:
        parts.append(prompt.strip())
    if prompt_file is not None:
        file_text = read_text_file(prompt_file)
        if file_text:
            parts.append(file_text)
    return "\n".join(parts) if parts else None


def load_serialized_file(path: Path) -> Any:
    text = path.read_text(encoding="utf-8")
    if not text.strip():
        return None
    if path.suffix.lower() == ".json":
        return json.loads(text)
    return yaml.safe_load(text)


def load_replacements(path: Path | None) -> ReplacementMap:
    if path is None:
        return {}

    data = load_serialized_file(path)
    if data is None:
        return {}

    if not isinstance(data, dict):
        raise ValueError("Replacement map must be a mapping or contain a replacements list.")

    if "replacements" in data:
        if set(data) != {"replacements"}:
            raise ValueError("Replacement map with 'replacements' can only contain that key.")
        replacements = data["replacements"]
        if not isinstance(replacements, list):
            raise ValueError("Replacement map 'replacements' must be a list.")
        result: ReplacementMap = {}
        for item in replacements:
            if not isinstance(item, dict) or "from" not in item or "to" not in item:
                raise ValueError("Each replacement item must contain 'from' and 'to'.")
            if not isinstance(item["from"], str) or not isinstance(item["to"], str):
                raise ValueError("Replacement source and target must be strings.")
            source = item["from"].strip()
            if not source:
                raise ValueError("Replacement source cannot be empty source.")
            if source in result:
                raise ValueError("Replacement source cannot be duplicate.")
            result[source] = item["to"]
        return result

    result: ReplacementMap = {}
    for source_value, target in data.items():
        if not isinstance(source_value, str) or not isinstance(target, str):
            raise ValueError("Replacement source and target must be strings.")
        source = source_value.strip()
        if not source:
            raise ValueError("Replacement source cannot be empty source.")
        if source in result:
            raise ValueError("Replacement source cannot be duplicate.")
        result[source] = target
    return result


def replace_text(text: str, replacements: ReplacementMap) -> str:
    for source, target in replacements.items():
        text = text.replace(source, target)
    return text


def apply_replacements_to_transcript(transcript: dict[str, Any], replacements: ReplacementMap) -> None:
    if not replacements:
        return

    for segment in transcript.get("segments", []):
        if not isinstance(segment, dict):
            continue
        if isinstance(segment.get("text"), str):
            segment["text"] = replace_text(segment["text"], replacements)
        for word in segment.get("words", []):
            if isinstance(word, dict) and isinstance(word.get("word"), str):
                word["word"] = replace_text(word["word"], replacements)
