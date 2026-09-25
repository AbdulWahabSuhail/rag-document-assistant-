# RAG Document Assistant

A lightweight retrieval-augmented document assistant that indexes local text/Markdown documents and returns the most relevant passages with source references.

The default implementation deliberately keeps retrieval local and deterministic so it runs without a paid API key. An LLM can be added as a generation layer later while preserving the retrieval and citation pipeline.

## Stack
Python, scikit-learn TF-IDF, cosine similarity, FastAPI, Pydantic

## Run
```bash
pip install -r requirements.txt
uvicorn app.main:app --reload
```

POST `/ask`
```json
{"question":"What is the refund policy?"}
```

## Architecture
1. Load local knowledge documents.
2. Split them into overlapping chunks.
3. Build TF-IDF vectors.
4. Vectorize the question.
5. Rank chunks by cosine similarity.
6. Return grounded passages and source names.

## Why this design?
For a portfolio project, deterministic retrieval makes the grounding layer easy to test and explain. A production extension could use sentence-transformer embeddings, a vector database, reranking, and an LLM generation step.

## Interview talking points
- What problem does RAG solve?
- How do chunk size and overlap affect retrieval?
- How would you evaluate retrieval quality?
- Why should generated answers preserve citations?
- When would a vector database become useful?
