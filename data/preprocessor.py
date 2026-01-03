# Normalisierung, Chunking

class Preprocessor:
    def process(self, raw_data: dict) -> str:
        texts = []
        for source, entries in raw_data.items():
            for e in entries:
                texts.append(f"[{source.upper()}] {e}")
        return "\n".join(texts)
