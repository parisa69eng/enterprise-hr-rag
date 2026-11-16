import json
import math
import re
from collections import Counter
from pathlib import Path

from src.vector_store import COLLECTION_NAME, get_qdrant_client


PROJECT_ROOT = Path(__file__).resolve().parent.parent
EVALUATION_FILE = (
    PROJECT_ROOT / "data" / "evaluation" / "retrieval_questions.json"
)

K1 = 1.5
B = 0.75


def tokenize(text: str) -> list[str]:
    """Convert text into lowercase word tokens."""
    return re.findall(r"\b[a-z0-9]+\b", text.lower())


def load_documents() -> list[dict]:
    """Load stored document chunks from the local Qdrant database."""
    client = get_qdrant_client()

    try:
        points, _ = client.scroll(
            collection_name=COLLECTION_NAME,
            limit=1000,
            with_payload=True,
            with_vectors=False,
        )
    finally:
        client.close()

    documents = []

    for point in points:
        payload = point.payload or {}
        text = payload.get("text", "")

        if not text:
            continue

        documents.append(
            {
                "text": text,
                "source": payload.get("source", ""),
                "section_title": payload.get("section_title", ""),
                "tokens": tokenize(text),
            }
        )

    return documents


def bm25_scores(
    query: str,
    documents: list[dict],
) -> list[float]:
    """Calculate BM25 scores using only Python's standard library."""
    if not documents:
        return []

    query_tokens = tokenize(query)
    document_count = len(documents)

    document_lengths = [
        len(document["tokens"]) for document in documents
    ]
    average_length = (
        sum(document_lengths) / document_count
        if document_count
        else 0
    )

    document_frequencies = Counter()

    for document in documents:
        document_frequencies.update(set(document["tokens"]))

    scores = []

    for document, document_length in zip(
        documents, document_lengths
    ):
        term_frequencies = Counter(document["tokens"])
        score = 0.0

        for term in query_tokens:
            frequency = term_frequencies[term]

            if frequency == 0:
                continue

            df = document_frequencies[term]

            inverse_document_frequency = math.log(
                1
                + (
                    document_count - df + 0.5
                )
                / (df + 0.5)
            )

            length_normalizer = (
                frequency
                + K1
                * (
                    1
                    - B
                    + B
                    * document_length
                    / average_length
                )
            ) if average_length else frequency + K1

            score += (
                inverse_document_frequency
                * frequency
                * (K1 + 1)
                / length_normalizer
            )

        scores.append(score)

    return scores


def evaluate_bm25(limit: int = 3) -> None:
    questions = json.loads(
        EVALUATION_FILE.read_text(encoding="utf-8")
    )
    documents = load_documents()

    if not documents:
        print("No document chunks found in the local Qdrant database.")
        return

    print(f"Loaded document chunks: {len(documents)}")

    hit_at_1 = 0
    hit_at_limit = 0
    reciprocal_rank_sum = 0.0

    for index, item in enumerate(questions, start=1):
        scores = bm25_scores(item["question"], documents)

        ranked_indices = sorted(
            range(len(documents)),
            key=lambda i: scores[i],
            reverse=True,
        )

        ranked_documents = [
            (documents[i], scores[i])
            for i in ranked_indices[:limit]
        ]

        expected_source = item["expected_source"]
        expected_section = item["expected_section"]

        matching_ranks = [
            rank
            for rank, (document, _) in enumerate(
                ranked_documents, start=1
            )
            if (
                document["source"] == expected_source
                and document["section_title"] == expected_section
            )
        ]

        first_match_rank = (
            matching_ranks[0] if matching_ranks else None
        )

        if first_match_rank == 1:
            hit_at_1 += 1

        if first_match_rank is not None:
            hit_at_limit += 1
            reciprocal_rank_sum += 1 / first_match_rank

        print(f"\nQuestion {index}: {item['question']}")
        print(f"Expected: {expected_source} -> {expected_section}")

        if first_match_rank is None:
            print(f"Result: MISS (not found in top {limit})")
        else:
            print(f"Result: MATCH at rank {first_match_rank}")

        for rank, (document, score) in enumerate(
            ranked_documents, start=1
        ):
            print(
                f"  {rank}. {document['source']} "
                f"| {document['section_title']} "
                f"| score={score:.4f}"
            )

    total = len(questions)

    print("\n========== BM25 Evaluation Summary ==========")
    print(f"Total questions: {total}")
    print(f"Document chunks: {len(documents)}")

    if total == 0:
        print("No evaluation questions found.")
        return

    print(f"Hit Rate@1: {hit_at_1 / total:.2%}")
    print(f"Hit Rate@{limit}: {hit_at_limit / total:.2%}")
    print(f"MRR@{limit}: {reciprocal_rank_sum / total:.4f}")


if __name__ == "__main__":
    evaluate_bm25()
