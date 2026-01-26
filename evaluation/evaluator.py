import re
from difflib import SequenceMatcher

import nltk.translate.meteor_score
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer


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
