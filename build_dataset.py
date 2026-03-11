from dotenv import load_dotenv
import os
import datasets

from data.commit_chronicle_extractor import CommitChronicleExtractor
from data.extractor import DataExtractor


def main():
    load_dotenv()
    token = os.getenv("GITHUB_TOKEN")

    api = DataExtractor()
    cc_extractor = CommitChronicleExtractor(api, split=datasets.Split.VALIDATION)

    cc_extractor.build_dataset(
        output_path="dataset.jsonl.zst",
        token=token
    )


if __name__ == "__main__":
    main()
