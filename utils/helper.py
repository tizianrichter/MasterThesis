import argparse
import yaml

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


def load_runs(config_path="repos.yaml"):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config["repos"]
