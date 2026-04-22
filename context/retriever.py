import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from data.ds_preprocessor import Preprocessor


class Retriever:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)
        self.database = []
        self.preprocessor = Preprocessor()

    def add_to_index(self, commits: list, release_notes: str, v_source: str, v_target: str, related_items: str):
        commit_texts = self.preprocessor.process(commits)
        embedding = self.model.encode([commit_texts])[0]
        self.database.append({
            "commits": commit_texts,
            "release_notes": release_notes,
            "v_source": v_source,
            "v_target": v_target,
            "related_items": related_items,
            "embedding": embedding
        })

    def query(self, current_commits: list):
        if not self.database:
            return None

        current_text = " ".join([c["message"] for c in current_commits])
        query_embedding = self.model.encode([current_text])[0]

        best_score = -1
        best_match = None

        for entry in self.database:
            score = cosine_similarity([query_embedding], [entry["embedding"]])[0][0]
            if score > best_score:
                best_score = score
                best_match = entry

        return best_match