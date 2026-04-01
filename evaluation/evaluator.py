import json
import re
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer

from llm.cloud_llm import CloudLLM


class ReleaseEvaluator:
    """
    Evaluates similarity between generated and human-written release notes
    using BLEU, ROUGE-L, and METEOR metrics.
    """

    def __init__(self, llm=None):
        self.rouge = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
        self.smooth = SmoothingFunction().method1
        self.llm = llm

    def extract_claims(self, text: str) -> list[str]:
        """
        Extract bullet-point claims from a block of text.

        Args:
            text: Input text containing bullet points.

        Returns:
            List of cleaned claim strings.
        """
        claims = []
        for line in text.splitlines():
            line = line.strip()
            if line.startswith(("-", "•")):
                claims.append(line.lstrip("-• ").strip())
        return claims

    def _bleu(self, a: str, b: str) -> float:
        """
        Compute BLEU-4 score between two strings.
        """
        return corpus_bleu(
            [[b.lower().split()]],
            [a.lower().split()],
            smoothing_function=self.smooth
        )

    def _rougeL(self, a: str, b: str) -> float:
        """
        Compute ROUGE-L score between two strings.
        """
        return self.rouge.score(a, b)['rougeL'].fmeasure

    def _meteor(self, a: str, b: str) -> float:
        """
        Compute METEOR score between two strings.
        """
        return meteor_score([b.split()], a.split())

    def similarity(self, a: str, b: str) -> dict:
        """
        Compute all similarity metrics between two strings.
        """
        return {
            "bleu4": self._bleu(a, b),
            "rougeL": self._rougeL(a, b),
            "meteor": self._meteor(a, b),
        }

    def match_claims(self, gt_list, pred_list):
        """
        Match ground truth claims to predicted claims
        based on highest average similarity.
        """
        matches = []
        used_pred = set()

        for gt in gt_list:
            best_score = 0
            best_idx = None
            best_metrics = None

            for i, pred in enumerate(pred_list):
                if i in used_pred:
                    continue

                sim = self.similarity(gt, pred)
                score = np.mean(list(sim.values()))

                if score > best_score:
                    best_score = score
                    best_idx = i
                    best_metrics = sim

            if best_idx is not None:
                used_pred.add(best_idx)
                matches.append((gt, pred_list[best_idx], best_metrics))

        return matches

    def compute_metrics(self, ground_truth, predictions):
        """
        Compute average BLEU, ROUGE-L, and METEOR scores
        over matched claim pairs.
        """
        matches = self.match_claims(ground_truth, predictions)

        if not matches:
            return {"bleu4": 0, "rougeL": 0, "meteor": 0}

        bleu_scores = []
        rouge_scores = []
        meteor_scores = []

        for gt, pred, sim in matches:
            bleu_scores.append(sim["bleu4"])
            rouge_scores.append(sim["rougeL"])
            meteor_scores.append(sim["meteor"])

        return {
            "bleu4": np.mean(bleu_scores),
            "rougeL": np.mean(rouge_scores),
            "meteor": np.mean(meteor_scores),
        }

    def corpus_scores(self, ground_truth, predictions):
        """
        Compute corpus-level BLEU, ROUGE-L, and METEOR scores
        using aligned matched claim pairs.
        """
        if not ground_truth or not predictions:
            return {"bleu4": 0, "rougeL": 0, "meteor": 0}

        matches = self.match_claims(ground_truth, predictions)

        if not matches:
            return {"bleu4": 0, "rougeL": 0, "meteor": 0}

        refs = []
        hyps = []
        rouge_scores = []
        meteor_scores = []

        for gt, pred, _ in matches:
            refs.append([gt.split()])
            hyps.append(pred.split())

            rouge_scores.append(self._rougeL(gt, pred))
            meteor_scores.append(self._meteor(gt, pred))

        bleu = corpus_bleu(refs, hyps, smoothing_function=self.smooth)
        rougeL = np.mean(rouge_scores)
        meteor = np.mean(meteor_scores)

        return {
            "bleu4": bleu,
            "rougeL": rougeL,
            "meteor": meteor
        }

    def evaluate_quality(self, generated_text: str) -> dict:
            """
            Evaluate clarity, conciseness, and organization of generated notes.
            """

            if not self.llm:
                return {}

            prompt = f"""
    You are an expert software documentation reviewer.

    Evaluate the following release notes on a scale from 1 to 5:

    - Clarity (easy to understand)
    - Conciseness (no unnecessary verbosity)
    - Organization (well structured and scannable)

    Also provide a short justification for each.

    Return ONLY valid JSON in this format:
    {{
      "clarity": {{"score": int, "reason": str}},
      "conciseness": {{"score": int, "reason": str}},
      "organization": {{"score": int, "reason": str}}
    }}

    Release Notes:
    \"\"\"
    {generated_text}
    \"\"\"
    """

            response = self.llm.generate(prompt, temperature=0.0, top_p=1.0)

            try:
                return json.loads(response)
            except:
                return {"error": response}

    def evaluate_content(self, generated_text: str, human_text: str) -> dict:
        """
        Compare generated notes to human notes for coverage and hallucinations.
        """

        if not self.llm:
            return {}

        prompt = f"""
    You are an expert evaluator of release notes.

    Compare the GENERATED release notes to the HUMAN reference.

    Evaluate:

    1. Coverage (1-5): Are important points from the human notes missing?
    2. Hallucination (1-5): Does the generated text include incorrect or extra information?

    Also list:
    - Missing points
    - Extra / hallucinated points

    Return ONLY valid JSON:

    {{
      "coverage": {{"score": int, "reason": str}},
      "hallucination": {{"score": int, "reason": str}},
      "missing_points": [str],
      "extra_points": [str]
    }}

    GENERATED:
    \"\"\"
    {generated_text}
    \"\"\"

    HUMAN:
    \"\"\"
    {human_text}
    \"\"\"
    """

        response = self.llm.generate(prompt, temperature=0.0, top_p=1.0)

        try:
            return json.loads(response)
        except:
            return {"error": response}


def extract_sections(text):
    """
    Extract generated and human release note sections from raw log text.
    """
    gen = re.search(r"GENERATED RELEASE NOTES:(.*?)HUMAN RELEASE NOTES:", text, re.DOTALL)
    human = re.search(r"HUMAN RELEASE NOTES:(.*)", text, re.DOTALL)

    return (
        gen.group(1).strip() if gen else "",
        human.group(1).strip() if human else ""
    )


def load_evaluations(logs_dir="logs", save_csv=False, csv_path="evaluation_one_shot.csv"):
    """
    Load evaluation data from log files and compute metrics.
    """
    #TODO: GPT API KEY, work key?
    llm = CloudLLM(model_name="gpt-4o-mini", api_key="")
    evaluator = ReleaseEvaluator(llm=llm)
    results = []

    for root, _, files in os.walk(logs_dir):
        for file in files:
            if not file.endswith(".txt"):
                continue

            path = os.path.join(root, file)

            with open(path, "r", encoding="utf-8") as f:
                content = f.read()

            gen_text, human_text = extract_sections(content)

            if not gen_text or not human_text:
                continue

            pred_claims = evaluator.extract_claims(gen_text)
            gt_claims = evaluator.extract_claims(human_text)

            metrics = evaluator.compute_metrics(gt_claims, pred_claims)
            corpus = evaluator.corpus_scores(gt_claims, pred_claims)

            parts = root.split(os.sep)

            quality = evaluator.evaluate_quality(gen_text)
            content_eval = evaluator.evaluate_content(gen_text, human_text)

            results.append({
                "project": parts[1] if len(parts) > 1 else "",
                "model": parts[3] if len(parts) > 3 else "",

                "avg_bleu4": metrics["bleu4"],
                "avg_rougeL": metrics["rougeL"],
                "avg_meteor": metrics["meteor"],

                "corpus_bleu4": corpus["bleu4"],
                "corpus_rougeL": corpus["rougeL"],
                "corpus_meteor": corpus["meteor"],

                "clarity": quality.get("clarity", {}).get("score"),
                "conciseness": quality.get("conciseness", {}).get("score"),
                "organization": quality.get("organization", {}).get("score"),

                "coverage": content_eval.get("coverage", {}).get("score"),
                "hallucination": content_eval.get("hallucination", {}).get("score"),
            })
            break

    df = pd.DataFrame(results)

    if save_csv:
        df.to_csv(csv_path, index=False)
        print(f"Saved to {csv_path}")

    return df


def aggregate_results(df, group_by=("project",)):
    """
    Aggregate evaluation results by computing mean metrics per group.
    """
    return df.groupby(list(group_by)).mean(numeric_only=True).reset_index()


def plot_metrics(df, group_by=("project",)):
    """
    Plot average BLEU, ROUGE-L, and METEOR scores.
    """
    metrics = ["avg_bleu4", "avg_rougeL", "avg_meteor"]

    plt.figure(figsize=(14, 6))

    if len(group_by) == 1:
        x = df[group_by[0]]
    else:
        x = df[group_by].astype(str).agg(" | ".join, axis=1)

    for m in metrics:
        plt.plot(x, df[m], marker='o', label=m)

    plt.xticks(rotation=45, ha='right')
    plt.title("Average Similarity Metrics")
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    df = load_evaluations("logs", save_csv=True)

    df_grouped = aggregate_results(df, group_by=("project",))
    plot_metrics(df_grouped)