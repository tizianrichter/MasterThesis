from collections import defaultdict

from data.extractor import DataExtractor
from data.ds_preprocessor import Preprocessor
from context.retriever import Retriever
from llm.local_llm import LocalLLM
from generation.release_notes import ReleaseNoteGenerator
from postprocess.formatter import Formatter
import os
from dotenv import load_dotenv
from evaluation.evaluator import ReleaseEvaluator
from utils.logging import redirect_output_per_run
import utils.helper as helper


def run_pipeline(
        repo_owner,
        repo_name,
        v_source,
        v_target,
        project_context,
        commits,
        release_notes,
        ex_v_source,
        ex_v_target,
        ex_commits,
        ex_release_notes,
        current_llm_model,
        prompt_only=False,
):
    # Logging
    log_path = redirect_output_per_run(
        repo_owner=repo_owner,
        repo_name=repo_name,
        model_name=current_llm_model,
        v_source=v_source,
        v_target=v_target,
    )
    print(f"Logging to: {log_path}\n")
    print("\n" + "=" * 80)
    print(
        f"Running pipeline for "
        f"{repo_owner}/{repo_name} "
        f"{v_source} → {v_target}"
    )
    print("=" * 80 + "\n")

    # Print variables
    all_vars = {
        "repo_owner": repo_owner,
        "repo_name": repo_name,
        "v_source": v_source,
        "v_target": v_target,
        "project_context": project_context,
        "current_llm_model": current_llm_model,
        "OLLAMA_NUM_THREADS": os.getenv("OLLAMA_NUM_THREADS"),
    }
    helper.print_variables(all_vars)

    # Initialize components
    preprocessor = Preprocessor()
    llm = LocalLLM(model_name=current_llm_model)
    generator = ReleaseNoteGenerator(llm)
    formatter = Formatter()
    evaluator = ReleaseEvaluator()

    cleaned_commits = preprocessor.process(commits)
    ex_cleaned_commits = preprocessor.process(ex_commits)
    prompt = generator.build_prompt(
        cleaned_commits, v_source, v_target, ex_cleaned_commits, ex_v_source, ex_v_target, ex_release_notes
    )

    print("PROMPT:\n" + prompt)

    if prompt_only:
        return

    generated_release_notes = generator.generate(prompt, temperature=0.1, top_p=0.9)
    final_output = formatter.format(generated_release_notes)

    print("\nGENERATED RELEASE NOTES:\n" + final_output)
    print("\nHUMAN RELEASE NOTES:\n" + release_notes)


def main():
    args = helper.parse_args()

    all_llm_models = ["smollm2:135m", "llama2:7b", "qwen2.5:7b-instruct"]
    current_llm_model = all_llm_models[2]

    releases = helper.load_releases_jsonl("dataset.jsonl.zst")

    # Group releases by repo_owner
    repos = defaultdict(list)
    for r in releases:
        repos[r["repo_owner"]].append(r)

    for repo_owner, repo_releases in repos.items():

        # Need at least 2 releases for in-context prompting
        if len(repo_releases) < 2:
            continue

        # Sort releases if you have version ordering (optional but recommended)
        # repo_releases = sorted(repo_releases, key=lambda x: x["v_source"])

        # Pick:
        example_release = repo_releases[0]
        target_release = repo_releases[1]

        run_pipeline(
            repo_owner=target_release["repo_owner"],
            repo_name=target_release["repo_name"],
            v_source=target_release["v_source"],
            v_target=target_release["v_target"],
            commits=target_release["commits"],
            project_context=target_release["project_context"],
            release_notes=target_release["release_notes"],

            ex_v_source=example_release["v_source"],
            ex_v_target=example_release["v_target"],
            ex_commits=example_release["commits"],
            ex_release_notes=example_release["release_notes"],

            current_llm_model=current_llm_model,
            prompt_only=args.prompt_only
        )


if __name__ == "__main__":
    main()
