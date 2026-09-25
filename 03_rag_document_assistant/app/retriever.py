from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class Retriever:
    def __init__(self, docs_dir):
        self.docs_dir = Path(docs_dir)
        self.chunks = []
        for p in sorted(self.docs_dir.glob("*")):
            if p.suffix.lower() not in {".txt",".md"}: continue
            text = p.read_text(encoding="utf-8")
            words = text.split()
            size, overlap = 90, 20
            step = size - overlap
            for i in range(0, len(words), step):
                chunk = " ".join(words[i:i+size])
                if chunk:
                    self.chunks.append({"source": p.name, "text": chunk})
        if not self.chunks:
            raise ValueError("No documents found")
        self.vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1,2))
        self.matrix = self.vectorizer.fit_transform([c["text"] for c in self.chunks])

    def search(self, question, top_k=3):
        q = self.vectorizer.transform([question])
        scores = cosine_similarity(q, self.matrix)[0]
        ids = scores.argsort()[::-1][:top_k]
        return [
            {"source": self.chunks[i]["source"], "score": round(float(scores[i]),4),
             "text": self.chunks[i]["text"]}
            for i in ids if scores[i] > 0
        ]
