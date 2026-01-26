import re


class Preprocessor:
    def __init__(self):
        self.merge_commit = re.compile(r"^merge\b", re.IGNORECASE)
        self.version_bump = re.compile(r"bump\s+version|release\s+v?\d", re.IGNORECASE)
        self.ci_only = re.compile(r"\b(ci|build|chore|deps)\b", re.IGNORECASE)

    def process(self, artifacts: str) -> str:
        """
        Preprocess a newline-separated artifact string and return
        a cleaned newline-separated string.
        """
        lines = artifacts.splitlines()
        cleaned_lines = []

        for line in lines:
            line = line.strip()
            if not line:
                continue

            if self._is_noise(line):
                continue

            line = self._normalize(line)
            cleaned_lines.append(line)

        return self._deduplicate(cleaned_lines)

    def _is_noise(self, text: str) -> bool:
        return any(
            pattern.search(text)
            for pattern in (
                self.merge_commit,
                self.version_bump,
                self.ci_only,
            )
        )

    def _normalize(self, text: str) -> str:
        text = text.rstrip(". ")

        if text:
            text = text[0].upper() + text[1:]

        text = self._normalize_tense(text)
        return text

    def _normalize_tense(self, text: str) -> str:
        replacements = {
            r"^add\b": "Added",
            r"^fix\b": "Fixed",
            r"^remove\b": "Removed",
            r"^update\b": "Updated",
            r"^improve\b": "Improved",
            r"^refactor\b": "Refactored",
        }

        for pattern, replacement in replacements.items():
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)

        return text

    def _deduplicate(self, lines: list[str]) -> str:
        seen = set()
        unique = []

        for line in lines:
            key = line.lower()
            if key not in seen:
                seen.add(key)
                unique.append(line)

        return "\n".join(unique)
