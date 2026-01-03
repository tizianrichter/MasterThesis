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
