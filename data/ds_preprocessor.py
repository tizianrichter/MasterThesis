import re
from collections import defaultdict
from data.semantic_grouper import SemanticGrouper

class Preprocessor:
    def __init__(self):
        self.merge_commit = re.compile(r"^merge", re.IGNORECASE)
        self.version_bump = re.compile(r"bump\s+version|release\s+v?\d", re.IGNORECASE)
        self.ci_only = re.compile(r"(ci|build|chore|deps)", re.IGNORECASE)

        self.ignore_paths = [
            "test",
            "docs",
            ".github",
            "readme",
        ]

        self.category_patterns = {
            "Added": [r"^add", r"^implement", r"^enable", r"^support", r"^initial commit"],
            "Fixed": [r"^fix", r"^bugfix", r"^correct", r"^resolve"],
            "Changed": [r"^update", r"^improve", r"^refactor", r"^rewrite", r"^modify", r"^rename", r"^change"],
            "Removed": [r"^remove", r"^delete"],
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
        grouper = SemanticGrouper()
        clusters = grouper.group(cleaned)
        summarized = ["; ".join(cluster) for cluster in clusters]
        grouped = self._categorize(summarized)
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
            r"^add": "Added",
            r"^fix": "Fixed",
            r"^remove": "Removed",
            r"^update": "Updated",
            r"^improve": "Improved",
            r"^refactor": "Refactored",
            r"^rewrite": "Rewrote",
            r"^modify": "Modified",
            r"^rename": "Renamed",
            r"^implement": "Implemented",
            r"^enable": "Enabled",
            r"^support": "Added support for",
            r"^initial commit": "Initial commit",
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