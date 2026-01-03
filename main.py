from data.extractor import DataExtractor
from data.preprocessor import Preprocessor
from context.retriever import Retriever
from llm.local_llm import LocalLLM
from generation.release_notes import ReleaseNoteGenerator
from postprocess.formatter import Formatter
import argparse


def main():
    parser = argparse.ArgumentParser(description="Example")
    parser.add_argument(
        "--prompt_only",
        action="store_true",
        help="If set, only generate the prompt without feeding into LLM"
    )
    args = parser.parse_args()

    llm_models = ["smollm2:135m", "llama2:7b"]
    v_source = "v0.13.4"
    v_target = "v0.13.5"
    project_context = "locally deployed AI model runner, designed to allow users to download and execute large language models (LLMs) directly on their personal computer"

    extractor = DataExtractor()
    preprocessor = Preprocessor()
    retriever = Retriever()
    llm = LocalLLM(model_name=llm_models[1])
    generator = ReleaseNoteGenerator(llm)
    formatter = Formatter()

    token = "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"

    commits = extractor.get_commits_between("ollama", "ollama", v_source, v_target, token=token)
    issues = extractor.get_issues(
        "ollama",
        "ollama",
        commits,
        token=token
    )
    # processed = preprocessor.process(raw_data)
    # enriched = retriever.enrich(issues)
    commits.extend(issues)
    extracted_data = "\n".join(commits)
    prompt = generator.build_prompt(extracted_data, v_source, v_target, project_context)
    print("PROMPT:" + prompt)
    if args.prompt_only:
        return
    release_notes = generator.generate(prompt, temperature=0.1, top_p=0.9)
    final_output = formatter.format(release_notes)

    print("RELEASE NOTES:" + final_output)


if __name__ == "__main__":
    main()
