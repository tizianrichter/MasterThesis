from data.extractor import DataExtractor
from data.preprocessor import Preprocessor
from context.retriever import Retriever
from llm.local_llm import LocalLLM
from generation.release_notes import ReleaseNoteGenerator
from postprocess.formatter import Formatter
import argparse
import os
from dotenv import load_dotenv
from evaluation.evaluator import ReleaseEvaluator
from utils.logging import redirect_output_per_run
import yaml
import utils.helper as helper

RUNS = [
    {
        "repo_owner": "ollama",
        "repo_name": "ollama",
        "v_source": "v0.13.4",
        "v_target": "v0.13.5",
        "project_context": (
            "locally deployed AI model runner, designed to allow users to "
            "download and execute large language models (LLMs) directly on their personal computer"
        ),
    },
    {
        "repo_owner": "psf",
        "repo_name": "requests",
        "v_source": "v2.31.0",
        "v_target": "v2.32.0",
        "project_context": "HTTP library for Python, focused on simplicity and readability",
    },
]


def run_pipeline(
        repo_owner,
        repo_name,
        v_source,
        v_target,
        project_context,
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
    ollama_num_threads = os.getenv("OLLAMA_NUM_THREADS")
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
    extractor = DataExtractor()
    preprocessor = Preprocessor()
    retriever = Retriever()
    llm = LocalLLM(model_name=current_llm_model)
    generator = ReleaseNoteGenerator(llm)
    formatter = Formatter()
    evaluator = ReleaseEvaluator()

    # Github Token
    load_dotenv()
    token = os.getenv("GITHUB_TOKEN")

    # Extract artifacts
    commits = extractor.get_commits_between(
        repo_owner, repo_name, v_source, v_target, token=token
    )
    issues = extractor.get_issues(repo_owner, repo_name, commits, token=token)
    commits.extend(issues)

    ground_truth_changes = commits + issues
    artifacts = "\n".join(commits)

    cleaned = preprocessor.process(artifacts)
    prompt = generator.build_prompt(
        artifacts, v_source, v_target, project_context
    )

    print("PROMPT:\n" + prompt)

    if prompt_only:
        return

    release_notes = generator.generate(prompt, temperature=0.1, top_p=0.9)
    final_output = formatter.format(release_notes)

    print("\nRELEASE NOTES:\n" + final_output)

    # Evaluation
    claims = evaluator.extract_claims(final_output)
    coverage = evaluator.coverage(ground_truth_changes, claims)
    hallucination = evaluator.hallucination_rate(ground_truth_changes, claims)

    print("\nEVALUATION:")
    print("\nCOVERAGE:")
    for k, v in coverage.items():
        print(f"{k}: {v:.2f}")

    print("\nHALLUCINATION RATE:")
    for k, v in hallucination.items():
        print(f"{k}: {v:.2f}")


def main():
    args = helper.parse_args()

    all_llm_models = ["smollm2:135m", "llama2:7b", "qwen2.5:7b-instruct"]
    current_llm_model = all_llm_models[2]

    for run in RUNS:
        run_pipeline(
            repo_owner=run["repo_owner"],
            repo_name=run["repo_name"],
            v_source=run["v_source"],
            v_target=run["v_target"],
            project_context=run["project_context"],
            current_llm_model=current_llm_model,
            prompt_only=args.prompt_only,
        )


if __name__ == "__main__":
    main()
