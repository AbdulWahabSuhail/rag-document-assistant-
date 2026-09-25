from pathlib import Path
from app.retriever import Retriever

def test_search(tmp_path):
    (tmp_path/"policy.txt").write_text("Refunds are available within 14 days for unused subscriptions.")
    r = Retriever(tmp_path)
    hits = r.search("When can I get a refund?")
    assert hits and hits[0]["source"] == "policy.txt"
