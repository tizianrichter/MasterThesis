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
import time


def run_pipeline(
        llm_mode,
        repo_owner,
        repo_name,
        v_source,
        v_target,
        project_context,
        commits,
        related_items,
        release_notes,
        ex_v_source1,
        ex_v_target1,
        ex_commits1,
        ex_release_notes1,
        ex_related_items1,
        ex_v_source2,
        ex_v_target2,
        ex_commits2,
        ex_release_notes2,
        ex_related_items2,
        current_llm_model,
        rag_context,
        prompt_only=False,
):
    time_start = time.perf_counter()

    # Logging
    log_path = redirect_output_per_run(
        llm_mode=llm_mode,
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

    cleaned_commits = preprocessor.process(commits)
    prompt = ""
    if llm_mode is helper.LLMModes.LLM_MODE_ZERO_SHOT:
        prompt = generator.build_prompt_zero_shot(
            cleaned_commits, v_source, v_target, project_context, related_items
        )
    elif llm_mode is helper.LLMModes.LLM_MODE_ONE_SHOT:
        ex_cleaned_commits1 = preprocessor.process(ex_commits1)
        prompt = generator.build_prompt_one_shot(
            cleaned_commits, v_source, v_target, project_context, related_items, ex_cleaned_commits1, ex_v_source1,
            ex_v_target1, ex_release_notes1, ex_related_items1,
        )
    elif llm_mode is helper.LLMModes.LLM_MODE_TWO_SHOT:
        ex_cleaned_commits1 = preprocessor.process(ex_commits1)
        ex_cleaned_commits2 = preprocessor.process(ex_commits2)
        prompt = generator.build_prompt_two_shot(
            cleaned_commits, v_source, v_target, project_context, related_items, ex_cleaned_commits1, ex_v_source1,
            ex_v_target1, ex_release_notes1, ex_related_items1,
            ex_cleaned_commits2, ex_v_source2, ex_v_target2, ex_release_notes2, ex_related_items2
        )
    elif llm_mode is helper.LLMModes.LLM_MODE_RAG:
        rag_artifacts = rag_context["commits"]
        rag_release_notes = rag_context["release_notes"]
        prompt = generator.build_prompt_rag(
            cleaned_commits, v_source, v_target, project_context, related_items, rag_artifacts, ex_v_source1,
            ex_v_target1, rag_release_notes, ex_related_items1)

    print("PROMPT:\n" + prompt)

    if prompt_only:
        return

    time_before_generation = time.perf_counter()

    generated_release_notes = generator.generate(prompt, temperature=0.1, top_p=0.9)
    final_output = formatter.format(generated_release_notes)

    print("\nGENERATED RELEASE NOTES:\n" + final_output)
    print("\nHUMAN RELEASE NOTES:\n" + release_notes)

    end_time = time.perf_counter()
    llm_duration = end_time - time_before_generation
    print(f"START TIME: {time_start:.2f}s\n")
    print(f"LLM DURATION: {llm_duration:.2f}s\n")
    print(f"END TIME: {end_time:.2f}s\n")


def main():
    args = helper.parse_args()

    all_llm_models = ["llama3.2:3b", "qwen2.5:7b-instruct", "llama3.1:8b", "mistral-small3.2:24b", "qwen3.5:35b", "llama3.3:70b"]
    #current_llm_model = all_llm_models[2]

    for llm_mode in helper.LLMModes:
        for current_llm_model in all_llm_models:

            releases = helper.load_releases_jsonl("dataset.jsonl.zst")

            # Group releases by repo_owner
            repos = defaultdict(list)
            for r in releases:
                repos[r["repo_owner"]].append(r)

            for repo_owner, repo_releases in repos.items():

                # Need at least 3 releases for in-context prompting
                if len(repo_releases) < 3:
                    continue

                # Sort releases
                repo_releases = sorted(repo_releases, key=lambda x: x["v_source"])

                # Pick:
                example_release1 = repo_releases[0]
                example_release2 = repo_releases[1]

                retriever = Retriever()
                retriever.add_to_index(example_release1["commits"], example_release1["release_notes"],
                                       example_release1["v_source"], example_release1["v_target"],
                                       example_release1["related_items"])
                retriever.add_to_index(example_release2["commits"], example_release2["release_notes"],
                                       example_release2["v_source"], example_release2["v_target"],
                                       example_release2["related_items"])

                for i in range(2, len(repo_releases)):
                    target_release = repo_releases[i]

                    print(f"\n---> Processing Release {i - 1}/{(len(repo_releases) - 2)} "
                          f"for {repo_owner}/{target_release['repo_name']}")

                    rag_context = retriever.query(target_release["commits"])

                    run_pipeline(
                        llm_mode=llm_mode,

                        repo_owner=target_release["repo_owner"],
                        repo_name=target_release["repo_name"],
                        v_source=target_release["v_source"],
                        v_target=target_release["v_target"],
                        commits=target_release["commits"],
                        related_items=target_release["related_items"],
                        project_context=target_release["project_context"],
                        release_notes=target_release["release_notes"],

                        ex_v_source1=example_release1["v_source"],
                        ex_v_target1=example_release1["v_target"],
                        ex_commits1=example_release1["commits"],
                        ex_release_notes1=example_release1["release_notes"],
                        ex_related_items1=example_release1["related_items"],

                        ex_v_source2=example_release2["v_source"],
                        ex_v_target2=example_release2["v_target"],
                        ex_commits2=example_release2["commits"],
                        ex_release_notes2=example_release2["release_notes"],
                        ex_related_items2=example_release2["related_items"],

                        current_llm_model=current_llm_model,
                        rag_context=rag_context,

                        prompt_only=args.prompt_only
                    )
                    retriever.add_to_index(target_release["commits"], target_release["release_notes"],
                                           target_release["v_source"], target_release["v_target"],
                                           target_release["related_items"])


if __name__ == "__main__":
    main()
