import re
from difflib import SequenceMatcher

import nltk.translate.meteor_score
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


class ReleaseEvaluator:
    def extract_claims(self, release_notes: str) -> list[str]:
        """
        Extract bullet-level claims from release notes
        """
        claims = []
        for line in release_notes.splitlines():
            line = line.strip()
            if line.startswith(("-", "•")):
                claims.append(line.lstrip("-• ").strip())
        return claims

    def _similar_bleu(self, a: str, b: str) -> float:
        """
        BLEU-4
        """
        a_tokens = a.lower().split()
        b_tokens = b.lower().split()
        return sentence_bleu([b_tokens], a_tokens, smoothing_function=SmoothingFunction().method1)

    def _similar_rouge_l(self, a: str, b: str) -> float:
        """
        ROUGE-L
        """
        scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
        scores = scorer.score(a, b)
        return scores['rougeL'].fmeasure

    def _similar_meteor(self, a: str, b: str) -> float:
        """
        METEOR
        """
        a_tokens = a.lower().split()
        b_tokens = b.lower().split()
        return meteor_score([b_tokens], a_tokens)

    def coverage(self, ground_truth: list[str], claims: list[str]) -> dict[str, float]:
        """
        Fraction of ground-truth items matched by at least one claim
        """
        metrics = {"bleu4": 0, "rougeL": 0, "meteor": 0}

        for gt in ground_truth:
            matched = {
                "bleu4": False,
                "rougeL": False,
                "meteor": False,
            }

            for c in claims:
                if self._similar_bleu(gt, c) > 0:
                    matched["bleu4"] = True
                if self._similar_rouge_l(gt, c) > 0:
                    matched["rougeL"] = True
                if self._similar_meteor(gt, c) > 0:
                    matched["meteor"] = True

            for k in metrics:
                if matched[k]:
                    metrics[k] += 1

        total = max(len(ground_truth), 1)
        return {k: v / total for k, v in metrics.items()}

    def hallucination_rate(self, ground_truth: list[str], claims: list[str]) -> dict[str, float]:
        """
        Fraction of claims not supported by any ground-truth item
        """
        metrics = {"bleu4": 0, "rougeL": 0, "meteor": 0}

        for c in claims:
            supported = {
                "bleu4": False,
                "rougeL": False,
                "meteor": False,
            }

            for gt in ground_truth:
                if self._similar_bleu(c, gt) > 0:
                    supported["bleu4"] = True
                if self._similar_rouge_l(c, gt) > 0:
                    supported["rougeL"] = True
                if self._similar_meteor(c, gt) > 0:
                    supported["meteor"] = True

            for k in metrics:
                if not supported[k]:
                    metrics[k] += 1

        total = max(len(claims), 1)
        return {k: v / total for k, v in metrics.items()}


def load_evaluations(logs_dir="logs", save_csv=False, csv_path="evaluation_summary.csv"):
    """Load evaluation metrics from all txt files into a DataFrame."""
    metric_pattern = re.compile(r"(bleu4|rougeL|meteor):\s*([\d.]+)")
    results = []

    for root, dirs, files in os.walk(logs_dir):
        for file in files:
            if file.endswith(".txt"):
                file_path = os.path.join(root, file)
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()

                coverage_match = re.search(r"COVERAGE:(.*?)HALLUCINATION RATE:", content, re.DOTALL)
                coverage = {}
                if coverage_match:
                    for m in metric_pattern.findall(coverage_match.group(1)):
                        coverage[m[0]] = float(m[1])

                parts = root.split(os.sep)
                project_name = parts[1] if len(parts) > 1 else ""
                subproject_name = parts[2] if len(parts) > 2 else ""
                model_name = parts[3] if len(parts) > 3 else ""

                results.append({
                    "project": project_name,
                    "subproject": subproject_name,
                    "model": model_name,
                    "coverage_bleu4": coverage.get("bleu4"),
                    "coverage_rougeL": coverage.get("rougeL"),
                    "coverage_meteor": coverage.get("meteor"),
                })

    df = pd.DataFrame(results)

    if save_csv:
        df.to_csv(csv_path, index=False)
        print(f"DataFrame saved as CSV at '{csv_path}'")

    return df


def plot_coverage_line(df):
    """Plot coverage metrics as a line plot."""
    projects = df['project']
    metrics = ['coverage_bleu4', 'coverage_rougeL', 'coverage_meteor']

    plt.figure(figsize=(14, 6))

    for metric in metrics:
        plt.plot(projects, df[metric], marker='o', label=metric.split('_')[1])

    plt.xlabel("Repo")
    plt.ylabel("Coverage Score")
    plt.title("Coverage Metrics per Project")
    plt.xticks(rotation=45, ha='right')
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    df = load_evaluations("logs", save_csv=False, csv_path="evaluation.csv")
    plot_coverage_line(df)
