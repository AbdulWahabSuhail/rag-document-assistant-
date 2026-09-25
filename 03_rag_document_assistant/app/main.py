from pathlib import Path
from fastapi import FastAPI
from pydantic import BaseModel, Field
from app.retriever import Retriever

DOCS = Path(__file__).resolve().parents[1]/"knowledge"
retriever = Retriever(DOCS)
app = FastAPI(title="RAG Document Assistant", version="1.0.0")

class Query(BaseModel):
    question: str = Field(min_length=3, max_length=1000)
    top_k: int = Field(default=3, ge=1, le=5)

@app.get("/health")
def health():
    return {"status":"ok","documents_indexed":len(retriever.chunks)}

@app.post("/ask")
def ask(q: Query):
    matches = retriever.search(q.question, q.top_k)
    return {"question":q.question,"sources":matches}
