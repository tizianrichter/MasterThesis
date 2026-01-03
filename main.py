from data.extractor import DataExtractor
from data.preprocessor import Preprocessor
from context.retriever import Retriever
from llm.local_llm import LocalLLM
from generation.release_notes import ReleaseNoteGenerator
from postprocess.formatter import Formatter


def main():
    llm_models = ["smollm2:135m", "llama2:7b"]
    v_source = "v0.13.4"
    v_target = "v0.13.5"

    extractor = DataExtractor()
    preprocessor = Preprocessor()
    retriever = Retriever()
    llm = LocalLLM(model_name=llm_models[0])
    generator = ReleaseNoteGenerator(llm)
    formatter = Formatter()

    token = "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"

    commits = extractor.get_commits_between("ollama", "ollama", v_source, v_target, token=token)
    issue_numbers = extractor.extract_issue_numbers(commits)
    issues = extractor.get_issues_by_numbers(
        "ollama",
        "ollama",
        issue_numbers,
        token=token
    )
    # processed = preprocessor.process(raw_data)
    # enriched = retriever.enrich(issues)
    commits.extend(issues)
    extracted_data = "\n".join(commits)
    release_notes = generator.generate(extracted_data, temperature=0.1, top_p=0.9, version_from=v_source,
                                       version_to=v_target,
                                       project_context="locally deployed AI model runner, designed to allow users to "
                                                       "download and execute large language models (LLMs) directly on "
                                                       "their personal computer")
    final_output = formatter.format(release_notes)

    print("RELEASE NOTES:" + final_output)


if __name__ == "__main__":
    main()
