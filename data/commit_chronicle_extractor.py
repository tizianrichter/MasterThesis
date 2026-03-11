from datasets import load_dataset
from collections import defaultdict
from typing import List, Dict, Optional
import datetime
import json
import zstandard as zstd



class CommitChronicleExtractor:
    """
    Build a clean, release-level dataset from CommitChronicle.

    Filters:
      - Stars >= min_stars
      - Contributors >= min_contributors
      - >= min_releases valid releases
      - >= min_commits_per_release
      - Non-trivial release notes
    """

    def __init__(self, data_extractor, split: str = "all"):
        print("Loading CommitChronicle dataset...")
        self.dataset = load_dataset(
            "JetBrains-Research/commit-chronicle",
            split=split
        )
        print(f"Loaded {len(self.dataset)} commits.\n")

        self.api = data_extractor
        self.repo_commits = self._group_commits_by_repo()

    def _group_commits_by_repo(self) -> Dict[str, List[Dict]]:
        repos = defaultdict(list)
        for row in self.dataset:
            if len(repos) >= 100:
                break
            repos[row["repo"]].append(row)

        print(f"Found {len(repos)} unique repositories.\n")
        return repos

    def _build_release_windows(
        self,
        repo: str,
        commits: List[Dict],
        tags: List[str],
        token: Optional[str],
        min_commits_per_release: int
    ) -> List[Dict]:

        owner, repo_name = repo.split("/")

        # Sort commits chronologically
        commits_sorted = sorted(
            commits,
            key=lambda x: datetime.datetime.strptime(
                x["date"], "%d.%m.%Y %H:%M:%S"
            )
        )

        releases = []

        # Iterate over consecutive tag pairs
        for i in range(1, len(tags)):
            base = tags[i]
            head = tags[i - 1]

            commit_hashes = self.api.get_commit_hashes_between(
                owner, repo_name, base, head, token
            )

            if not commit_hashes:
                continue

            window_commits = [
                c for c in commits_sorted if c["hash"] in commit_hashes
            ]

            if len(window_commits) < min_commits_per_release:
                continue

            release_notes = self.api.get_release_notes(
                owner, repo_name, head, token
            )

            if not release_notes:
                continue

            releases.append({
                "repo": repo,
                "release_tag": head,
                "base_tag": base,
                "num_commits": len(window_commits),
                "commits": [
                    {
                        "message": c["message"],
                        "diff": c["mods"]
                    }
                    for c in window_commits
                ],
                "release_notes": release_notes
            })

        return releases

    def build_dataset(
        self,
        output_path: str,
        token: Optional[str] = None,
        min_stars: int = 50,
        min_contributors: int = 10,
        min_releases: int = 3,
        min_commits_per_release: int = 20
    ):

        final_dataset = []

        for repo, commits in self.repo_commits.items():
            owner, repo_name = repo.split("/")

            print(f"Checking {repo}...")

            metadata = self.api.get_repo_metadata(
                owner, repo_name, token
            )
            if not metadata:
                continue

            if metadata.get("archived", False):
                continue

            if metadata.get("stargazers_count", 0) < min_stars:
                continue

            contributors = self.api.get_contributors_count(
                owner, repo_name, token
            )

            if contributors < min_contributors:
                continue

            tags = self.api.get_tags(
                owner, repo_name, token
            )

            if len(tags) < min_releases:
                continue

            releases = self._build_release_windows(
                repo,
                commits,
                tags,
                token,
                min_commits_per_release
            )

            if len(releases) < min_releases:
                continue

            final_dataset.extend(releases)

            print(f"  -> Added {len(releases)} releases.")

        print(f"\nFinal dataset size: {len(final_dataset)} releases")

        with open(output_path, "wb") as f:
            compressor = zstd.ZstdCompressor(level=10)
            with compressor.stream_writer(f) as writer:
                for item in final_dataset:
                    line = json.dumps(item) + "\n"
                    writer.write(line.encode("utf-8"))

        print(f"Saved to {output_path}")