import argparse
import io

import yaml
import json
import zstandard as zstd


def parse_args():
    parser = argparse.ArgumentParser(description="Generate release notes for a repo")
    parser.add_argument(
        "--prompt_only",
        action="store_true",
        help="If set, only generate the prompt without feeding into LLM"
    )
    return parser.parse_args()


def print_variables(vars_dict):
    for name, value in vars_dict.items():
        print(f"{name:<25} = {value}")
    print()


def load_repos_yaml(config_path="repos.yaml"):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config["repos"]


def load_releases_json(path: str):
    """
    Load the release-level JSONL dataset.
    Returns a list of dicts with keys:
    repo_owner, repo_name, v_source, v_target, project_context, commits, release_notes
    """
    releases = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            owner, name = item["repo"].split("/")
            releases.append({
                "repo_owner": owner,
                "repo_name": name,
                "v_source": item["base_tag"],
                "v_target": item["release_tag"],
                "project_context": "",  # optional: could extract from description or leave empty
                "commits": item["commits"],
                "release_notes": item["release_notes"]
            })
    return releases


def load_releases_jsonl(path: str):
    """
    Load the release-level JSONL dataset (.jsonl.zst).

    Returns a list of dicts with keys:
    repo_owner, repo_name, v_source, v_target, project_context, commits, release_notes
    """

    releases = []

    with open(path, "rb") as f:
        dctx = zstd.ZstdDecompressor()

        with dctx.stream_reader(f) as reader:
            text_stream = io.TextIOWrapper(reader, encoding="utf-8")

            for line in text_stream:
                item = json.loads(line)

                owner, name = item["repo"].split("/")

                releases.append({
                    "repo_owner": owner,
                    "repo_name": name,
                    "v_source": item["base_tag"],
                    "v_target": item["release_tag"],
                    "project_context": item["project_context"],
                    "commits": item["commits"],
                    "release_notes": item["release_notes"],
                    "related_items": item["related_items"]
                })

    return releases
