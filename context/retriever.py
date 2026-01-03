# RAG (Embedding + Suche)

class Retriever:
    def enrich(self, text: str) -> str:
        # MVP: keine echten Embeddings
        return (
            "Project context: Web backend service\n"
            "Target audience: End users\n\n"
            + text
        )
