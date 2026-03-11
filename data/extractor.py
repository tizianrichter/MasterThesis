# Git / Issues / PRs
import requests
from typing import List, Dict, Optional
import re
from typing import Set, List, Dict


class DataExtractor:

    def get_commits_between(
            self,
            owner: str,
            repo: str,
            base: str,
            head: str,
            token: Optional[str] = None
    ) -> List[str]:
        """
        Get all commits between two GitHub refs (tags, branches, or SHAs)
        and return formatted commit messages.

        :param owner: GitHub organization or username
        :param repo: Repository name
        :param base: Base tag/version/branch (older)
        :param head: Head tag/version/branch (newer)
        :param token: Optional GitHub personal access token
        :return: List of formatted commit messages
        """

        url = f"https://api.github.com/repos/{owner}/{repo}/compare/{base}...{head}"

        headers = {
            "Accept": "application/vnd.github+json"
        }

        if token:
            headers["Authorization"] = f"Bearer {token}"

        response = requests.get(url, headers=headers)
        response.raise_for_status()

        data = response.json()

        commits = data.get("commits", [])

        formatted_commits = []
        for commit in commits:
            message = commit["commit"]["message"].splitlines()[0].strip()
            formatted_commits.append(f"[COMMIT] {message}")

        return formatted_commits

    def extract_issue_numbers(self, commit_messages: List[str]) -> Set[int]:
        """
        Extract GitHub issue/PR numbers from formatted commit messages.

        Expected format:
        [COMMIT] <message>

        :param commit_messages: List of formatted commit messages
        :return: Set of issue/PR numbers
        """

        issue_numbers = set()
        pattern = re.compile(r"#(\d+)")

        for message in commit_messages:
            matches = pattern.findall(message)
            for match in matches:
                issue_numbers.add(int(match))

        return issue_numbers

    def get_issues(
            self,
            owner: str,
            repo: str,
            commits: List[str],
            token: Optional[str] = None
    ) -> List[str]:
        issue_numbers = self.extract_issue_numbers(commits)
        headers = {"Accept": "application/vnd.github+json"}
        if token:
            headers["Authorization"] = f"Bearer {token}"

        issues = []

        for number in sorted(issue_numbers):
            url = f"https://api.github.com/repos/{owner}/{repo}/issues/{number}"
            response = requests.get(url, headers=headers)

            if response.status_code != 200:
                continue

            data = response.json()

            if "pull_request" in data:
                title = data["title"]
                issues.append(f"[PULL REQUEST] {title}")
            else:
                title = data["title"]
                issues.append(f"[ISSUE] {title}")

        return issues

    def get_code_diff_between(
            self,
            owner: str,
            repo: str,
            base: str,
            head: str,
            token: Optional[str] = None,
            max_chars: int = 12000
    ) -> str:
        """
        Get unified diff between two GitHub refs using the compare API.

        :param owner: GitHub organization or username
        :param repo: Repository name
        :param base: Base tag/version/branch
        :param head: Head tag/version/branch
        :param token: Optional GitHub personal access token
        :param max_chars: Safety limit to avoid huge prompts
        :return: Unified diff string
        """

        url = f"https://api.github.com/repos/{owner}/{repo}/compare/{base}...{head}"

        headers = {
            "Accept": "application/vnd.github+json"
        }
        if token:
            headers["Authorization"] = f"Bearer {token}"

        response = requests.get(url, headers=headers)
        response.raise_for_status()
        data = response.json()

        diff_chunks = []

        for file in data.get("files", []):
            patch = file.get("patch")
            if not patch:
                continue

            filename = file.get("filename")
            diff_chunks.append(f"--- a/{filename}\n+++ b/{filename}\n{patch}")

        diff_text = "\n\n".join(diff_chunks)

        if len(diff_text) > max_chars:
            diff_text = diff_text[:max_chars] + "\n\n[DIFF TRUNCATED]"

        return diff_text

    def get_repo_metadata(
            self,
            owner: str,
            repo: str,
            token: Optional[str] = None
    ) -> Optional[Dict]:
        """
        Fetch repository metadata (stars, forks, archived status, etc.)
        """
        url = f"https://api.github.com/repos/{owner}/{repo}"
        headers = {"Accept": "application/vnd.github+json"}
        if token:
            headers["Authorization"] = f"Bearer {token}"

        response = requests.get(url, headers=headers)
        if response.status_code != 200:
            return None

        return response.json()

    def get_contributors_count(
            self,
            owner: str,
            repo: str,
            token: Optional[str] = None
    ) -> int:
        """
        Get approximate number of contributors.
        """
        url = f"https://api.github.com/repos/{owner}/{repo}/contributors?per_page=1&anon=true"
        headers = {"Accept": "application/vnd.github+json"}
        if token:
            headers["Authorization"] = f"Bearer {token}"

        response = requests.get(url, headers=headers)

        if response.status_code != 200:
            return 0

        # Use pagination header trick to estimate total
        if "Link" in response.headers:
            link = response.headers["Link"]
            if 'rel="last"' in link:
                last_page = int(link.split("page=")[-1].split(">")[0])
                return last_page

        return len(response.json())

    def get_tags(
            self,
            owner: str,
            repo: str,
            token: Optional[str] = None
    ) -> List[str]:
        """
        Fetch repository tags (max 100).
        """
        url = f"https://api.github.com/repos/{owner}/{repo}/tags?per_page=100"
        headers = {"Accept": "application/vnd.github+json"}
        if token:
            headers["Authorization"] = f"Bearer {token}"

        response = requests.get(url, headers=headers)
        if response.status_code != 200:
            return []

        return [tag["name"] for tag in response.json()]

    def get_release_notes(
            self,
            owner: str,
            repo: str,
            tag: str,
            token: Optional[str] = None,
            min_length: int = 30
    ) -> Optional[str]:
        """
        Fetch GitHub release notes for a specific tag.
        Filters out empty or trivial releases.
        """
        url = f"https://api.github.com/repos/{owner}/{repo}/releases/tags/{tag}"
        headers = {"Accept": "application/vnd.github+json"}
        if token:
            headers["Authorization"] = f"Bearer {token}"

        response = requests.get(url, headers=headers)

        if response.status_code != 200:
            return None

        body = response.json().get("body")

        if not body or len(body.strip()) < min_length:
            return None

        return body.strip()

    def get_commit_hashes_between(
            self,
            owner: str,
            repo: str,
            base: str,
            head: str,
            token: Optional[str] = None
    ) -> Set[str]:
        """
        Get commit SHAs between two refs.
        Useful for mapping to CommitChronicle commits.
        """
        url = f"https://api.github.com/repos/{owner}/{repo}/compare/{base}...{head}"
        headers = {"Accept": "application/vnd.github+json"}
        if token:
            headers["Authorization"] = f"Bearer {token}"

        response = requests.get(url, headers=headers)

        if response.status_code != 200:
            return set()

        data = response.json()
        return {commit["sha"] for commit in data.get("commits", [])}