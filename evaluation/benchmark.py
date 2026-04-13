# evaluation/benchmark.py
# Runs a fixed set of benchmark queries across text, table, and image modalities.
# Reports retrieval accuracy and answer quality metrics.

import json
import os
from retrieval.retriever import Retriever
from generation.generator import generate_answer

QUERIES_PATH = os.path.join(os.path.dirname(__file__), "queries.json")


def run_benchmark() -> None:
    with open(QUERIES_PATH, encoding="utf-8") as f:
        queries = json.load(f)

    retriever = Retriever()
    results_by_modality: dict[str, dict] = {}

    for entry in queries:
        qid = entry["id"]
        modality = entry["modality"]
        question = entry["question"]
        expected_page = entry.get("expected_page")

        if not question:
            print(f"[Query {qid}] Skipping — no question defined.")
            continue

        pages = retriever.retrieve(question)
        top1_page = pages[0].page_num if pages else None
        hit = (top1_page == expected_page) if expected_page is not None else None

        answer = generate_answer(question, pages)

        if modality not in results_by_modality:
            results_by_modality[modality] = {"total": 0, "hits": 0}
        results_by_modality[modality]["total"] += 1
        if hit:
            results_by_modality[modality]["hits"] += 1

        print(f"\n=== Query {qid} [{modality}] ===")
        print(f"Q: {question}")
        print(f"Top-1 retrieved page: {top1_page} | Expected: {expected_page} | Hit@1: {hit}")
        print(f"A: {answer}")

    print("\n=== Benchmark Summary ===")
    for modality, stats in results_by_modality.items():
        total = stats["total"]
        hits = stats["hits"]
        acc = hits / total if total > 0 else 0.0
        print(f"  {modality}: Hit@1 = {hits}/{total} ({acc:.0%})")
