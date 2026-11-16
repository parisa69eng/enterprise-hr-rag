
import json
from pathlib import Path

from src.evaluate_bm25 import (
    tokenize,
    bm25_scores,
    load_documents,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent
EVALUATION_FILE = (
    PROJECT_ROOT / "data" / "evaluation" / "retrieval_questions.json"
)

TITLE_BOOSTS = [0.5, 1.0, 2.0]


def calculate_title_score(
    query: str,
    section_title: str,
) -> float:
    query_tokens = set(tokenize(query))
    title_tokens = set(tokenize(section_title))

    if not title_tokens:
        return 0.0

    matching_tokens = query_tokens.intersection(title_tokens)

    return len(matching_tokens) / len(title_tokens)


def evaluate_hybrid(
    questions: list,
    documents: list,
    title_boost: float,
    limit: int = 3,
) -> dict:
    hit_at_1 = 0
    hit_at_limit = 0
    reciprocal_rank_sum = 0.0

    for item in questions:
        question = item["question"]
        content_scores = bm25_scores(question, documents)

        ranked_results = []

        for doc_index, document in enumerate(documents):
            title_score = calculate_title_score(
                question,
                document["section_title"],
            )

            hybrid_score = (
                content_scores[doc_index]
                + title_boost * title_score
            )

            ranked_results.append(
                (document, hybrid_score)
            )

        ranked_results.sort(
            key=lambda result: result[1],
            reverse=True,
        )

        top_results = ranked_results[:limit]

        matching_ranks = [
            rank
            for rank, (document, _) in enumerate(
                top_results,
                start=1,
            )
            if (
                document["source"] == item["expected_source"]
                and document["section_title"] == item["expected_section"]
            )
        ]

        if matching_ranks:
            rank = matching_ranks[0]

            hit_at_limit += 1
            reciprocal_rank_sum += 1 / rank

            if rank == 1:
                hit_at_1 += 1

    total = len(questions)

    return {
        "title_boost": title_boost,
        "hit_at_1": hit_at_1 / total if total else 0,
        "hit_at_limit": hit_at_limit / total if total else 0,
        "mrr": reciprocal_rank_sum / total if total else 0,
    }


def main() -> None:
    questions = json.loads(
        EVALUATION_FILE.read_text(encoding="utf-8")
    )

    documents = load_documents()

    if not documents:
        print("No document chunks found.")
        return

    if not questions:
        print("No evaluation questions found.")
        return

    print(f"Loaded document chunks: {len(documents)}")
    print(f"Evaluation questions: {len(questions)}")
    print("\nRunning experiments...\n")

    results = []

    for boost in TITLE_BOOSTS:
        result = evaluate_hybrid(
            questions=questions,
            documents=documents,
            title_boost=boost,
        )

        results.append(result)

    print("=" * 75)
    print("HYBRID SEARCH EXPERIMENT RESULTS")
    print("=" * 75)

    print(
        f"{'Title Boost':<15}"
        f"{'Hit Rate@1':<18}"
        f"{'Hit Rate@3':<18}"
        f"{'MRR@3':<12}"
    )

    print("-" * 75)

    for result in results:
        print(
            f"{result['title_boost']:<15.1f}"
            f"{result['hit_at_1']:<18.2%}"
            f"{result['hit_at_limit']:<18.2%}"
            f"{result['mrr']:<12.4f}"
        )

    print("=" * 75)


if __name__ == "__main__":
    main()

