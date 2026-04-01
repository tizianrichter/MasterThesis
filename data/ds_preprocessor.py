import re
from collections import defaultdict

class Preprocessor:
    def __init__(self):
        self.merge_commit = re.compile(r"^merge\b", re.IGNORECASE)
        self.version_bump = re.compile(r"bump\s+version|release\s+v?\d", re.IGNORECASE)
        self.ci_only = re.compile(r"\b(ci|build|chore|deps)\b", re.IGNORECASE)

        self.ignore_paths = [
            "test",
            "docs",
            ".github",
            "readme",
        ]

        self.category_patterns = {
            "Added": [r"^add\b", r"^implement\b", r"^enable\b", r"^support\b", r"^initial commit\b"],
            "Fixed": [r"^fix\b", r"^bugfix\b", r"^correct\b", r"^resolve\b"],
            "Changed": [r"^update\b", r"^improve\b", r"^refactor\b", r"^rewrite\b", r"^modify\b", r"^rename\b", r"^change\b"],
            "Removed": [r"^remove\b", r"^delete\b"],
            "Security": [r"security", r"vulnerability", r"cve"],
        }

        self.compiled_category_patterns = {
            cat: [re.compile(p, re.IGNORECASE) for p in patterns]
            for cat, patterns in self.category_patterns.items()
        }

    def process(self, commits: list[dict]) -> str:
        """
        Process commits and return grouped, release-note-ready CHANGES section
        """
        cleaned = []

        for commit in commits:
            message = commit.get("message", "").strip()
            if not message:
                continue
            if self._is_noise(message):
                continue
            if self._touches_only_docs(commit.get("diff", [])):
                continue

            message = self._normalize(message)
            if message:
                cleaned.append(message)

        cleaned = self._deduplicate(cleaned)
        grouped = self._categorize(cleaned)
        return self._format_grouped(grouped)

    def _categorize(self, messages: list[str]) -> dict[str, list[str]]:
        grouped = defaultdict(list)
        for msg in messages:
            assigned = False
            for category, patterns in self.compiled_category_patterns.items():
                if any(p.search(msg) for p in patterns):
                    grouped[category].append(msg)
                    assigned = True
                    break
            if not assigned:
                grouped["Changed"].append(msg)  # default
        return grouped

    def _format_grouped(self, grouped: dict[str, list[str]]) -> str:
        sections_order = ["Added", "Fixed", "Changed", "Removed", "Security"]
        lines = []
        for section in sections_order:
            items = grouped.get(section, [])
            if items:
                lines.append(f"{section}:")
                for i in items:
                    lines.append(f"- {i}")
                lines.append("")  # blank line between sections
        return "\n".join(lines).strip()

    def _normalize(self, text: str) -> str:
        text = text.splitlines()[0].strip()
        if not text:
            return ""

        text = re.sub(r"^[-*]\s*", "", text)
        text = re.sub(r"^\[.*?\]\s*", "", text)

        text = self._normalize_tense(text)

        if text and not text[0].isupper():
            text = text[0].upper() + text[1:]
        return text

    def _normalize_tense(self, text: str) -> str:
        replacements = {
            r"^add\b": "Added",
            r"^fix\b": "Fixed",
            r"^remove\b": "Removed",
            r"^update\b": "Updated",
            r"^improve\b": "Improved",
            r"^refactor\b": "Refactored",
            r"^rewrite\b": "Rewrote",
            r"^modify\b": "Modified",
            r"^rename\b": "Renamed",
            r"^implement\b": "Implemented",
            r"^enable\b": "Enabled",
            r"^support\b": "Added support for",
            r"^initial commit\b": "Initial commit",
        }
        for pattern, replacement in replacements.items():
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        return text

    def _is_noise(self, text: str) -> bool:
        return any(pattern.search(text) for pattern in (self.merge_commit, self.version_bump, self.ci_only))

    def _touches_only_docs(self, diffs: list[dict]) -> bool:
        for change in diffs:
            path = (change.get("new_path") or change.get("old_path") or "").lower()
            if not any(ignore in path for ignore in self.ignore_paths):
                return False
        return True

    def _deduplicate(self, messages: list[str]) -> list[str]:
        seen = set()
        unique = []
        for msg in messages:
            key = msg.lower()
            if key not in seen:
                seen.add(key)
                unique.append(msg)
        return unique