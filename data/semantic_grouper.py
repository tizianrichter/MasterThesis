from sklearn.cluster import AgglomerativeClustering
from sentence_transformers import SentenceTransformer
import numpy as np

class SemanticGrouper:
    def __init__(self, model_name="all-MiniLM-L6-v2", distance_threshold=0.6):
        self.model = SentenceTransformer(model_name)
        self.distance_threshold = distance_threshold

    def group(self, messages: list[str]) -> list[list[str]]:
        # embed messages
        embeddings = self.model.encode(messages, convert_to_tensor=False)

        # cluster similar messages
        clustering = AgglomerativeClustering(
            n_clusters=None,
            linkage='average',
            distance_threshold=self.distance_threshold
        )
        labels = clustering.fit_predict(embeddings)

        # group messages by cluster
        grouped = {}
        for label, msg in zip(labels, messages):
            grouped.setdefault(label, []).append(msg)

        return list(grouped.values())