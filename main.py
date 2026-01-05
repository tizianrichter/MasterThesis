from data.extractor import DataExtractor
from data.preprocessor import Preprocessor
from context.retriever import Retriever
from llm.local_llm import LocalLLM
from generation.release_notes import ReleaseNoteGenerator
from postprocess.formatter import Formatter
import argparse
import os
from dotenv import load_dotenv


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


def main():
    args = parse_args()

    # Configuration
    all_llm_models = ["smollm2:135m", "llama2:7b", "qwen2.5:7b-instruct"]
    current_llm_model = all_llm_models[2]
    v_source = "v0.13.4"
    v_target = "v0.13.5"
    project_context = (
        "locally deployed AI model runner, designed to allow users to "
        "download and execute large language models (LLMs) directly on their personal computer"
    )
    repo_owner = "ollama"
    repo_name = "ollama"

    # Print variables
    all_vars = {**locals(), "OLLAMA_NUM_THREADS": os.getenv("OLLAMA_NUM_THREADS")}
    print_variables(all_vars)

    # Initialize components
    extractor = DataExtractor()
    preprocessor = Preprocessor()
    retriever = Retriever()
    llm = LocalLLM(model_name=current_llm_model)
    generator = ReleaseNoteGenerator(llm)
    formatter = Formatter()

    # Github Token
    load_dotenv()
    token = os.getenv("GITHUB_TOKEN")

    # Extract artifacts
    commits = extractor.get_commits_between(repo_owner, repo_name, v_source, v_target, token=token)
    issues = extractor.get_issues(repo_owner, repo_name, commits, token=token)
    commits.extend(issues)
    code_diff = extractor.get_code_diff_between(repo_owner, repo_name, v_source, v_target, token=token)

    # Generate prompt and release notes
    artifacts = "\n".join(commits)
    prompt = generator.build_prompt(artifacts, v_source, v_target, project_context)
    print("PROMPT:\n" + prompt)

    if args.prompt_only:
        return

    release_notes = generator.generate(prompt, temperature=0.1, top_p=0.9)
    final_output = formatter.format(release_notes)
    print("\nRELEASE NOTES:\n" + final_output)


if __name__ == "__main__":
    main()
