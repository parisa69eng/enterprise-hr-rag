import json
from pathlib import Path

from src.embeddings import create_embedding
from src.vector_store import COLLECTION_NAME, get_qdrant_client


PROJECT_ROOT = Path(__file__).resolve().parent.parent
EVALUATION_FILE = (
    PROJECT_ROOT / "data" / "evaluation" / "retrieval_questions.json"
)


def evaluate_retrieval(limit: int = 3) -> None:
    questions = json.loads(
        EVALUATION_FILE.read_text(encoding="utf-8")
    )

    client = get_qdrant_client()

    hit_at_1 = 0
    hit_at_3 = 0
    reciprocal_rank_sum = 0.0

    try:
        for index, item in enumerate(questions, start=1):
            question = item["question"]
            expected_source = item["expected_source"]
            expected_section = item["expected_section"]

            query_vector = create_embedding(question)

            results = client.query_points(
                collection_name=COLLECTION_NAME,
                query=query_vector,
                limit=limit,
                with_payload=True,
            )

            points = results.points
            matching_ranks = []

            for rank, point in enumerate(points, start=1):
                payload = point.payload or {}

                is_match = (
                    payload.get("source") == expected_source
                    and payload.get("section_title") == expected_section
                )

                if is_match:
                    matching_ranks.append(rank)

            first_match_rank = (
                matching_ranks[0] if matching_ranks else None
            )

            if first_match_rank == 1:
                hit_at_1 += 1

            if first_match_rank is not None:
                hit_at_3 += 1
                reciprocal_rank_sum += 1 / first_match_rank

            print(f"\nQuestion {index}: {question}")
            print(
                f"Expected: {expected_source} "
                f"-> {expected_section}"
            )

            if first_match_rank is None:
                print(f"Result: MISS (not found in top {limit})")
            else:
                print(f"Result: MATCH at rank {first_match_rank}")

            for rank, point in enumerate(points, start=1):
                payload = point.payload or {}
                print(
                    f"  {rank}. {payload.get('source')} "
                    f"| {payload.get('section_title')} "
                    f"| score={point.score:.4f}"
                )

    finally:
        client.close()

    total = len(questions)

    if total == 0:
        print("\nNo evaluation questions found.")
        return

    print("\n========== Evaluation Summary ==========")
    print(f"Total questions: {total}")
    print(f"Hit Rate@1: {hit_at_1 / total:.2%}")
    print(f"Hit Rate@{limit}: {hit_at_3 / total:.2%}")
    print(f"MRR@{limit}: {reciprocal_rank_sum / total:.4f}")


if __name__ == "__main__":
    evaluate_retrieval()
