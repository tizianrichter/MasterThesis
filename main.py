from data.extractor import DataExtractor
from data.preprocessor import Preprocessor
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
        current_llm_model,
        release_notes,
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

    artifacts = "\n".join(commits)

    cleaned = preprocessor.process(artifacts)
    prompt = generator.build_prompt(
        artifacts, v_source, v_target, project_context
    )

    print("PROMPT:\n" + prompt)

    if prompt_only:
        return

    generated_release_notes = generator.generate(prompt, temperature=0.1, top_p=0.9)
    final_output = formatter.format(generated_release_notes)

    print("\nGENERATED RELEASE NOTES:\n" + final_output)

    # Evaluation
    claims = evaluator.extract_claims(final_output)
    coverage = evaluator.coverage(release_notes, claims)
    hallucination = evaluator.hallucination_rate(release_notes, claims)

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

    # repos = helper.load_repos_yaml()
    # for repo in repos:
    #   run_pipeline(
    #      repo_owner=repo["repo_owner"],
    #      repo_name=repo["repo_name"],
    #      v_source=repo["v_source"],
    #      v_target=repo["v_target"],
    #      project_context=repo["project_context"],
    #      current_llm_model=current_llm_model,
    #      prompt_only=args.prompt_only
    #  )

    releases = helper.load_releases_jsonl("dataset.jsonl.zst")

    repo_owner = "jupyterhub"

    for release in releases:
        if repo_owner == release["repo_owner"]:
            continue
        repo_owner = release["repo_owner"]
        run_pipeline(
            repo_owner=release["repo_owner"],
            repo_name=release["repo_name"],
            v_source=release["v_source"],
            v_target=release["v_target"],
            project_context=release["project_context"],
            current_llm_model=current_llm_model,
            release_notes=release["release_notes"],
            prompt_only=args.prompt_only
        )


if __name__ == "__main__":
    main()
