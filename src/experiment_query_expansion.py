from src.evaluate_bm25 import (
    bm25_scores,
    load_documents,
)

QUESTION = (
    "What proof must I provide for an expensive business purchase?"
)

EXPECTED_SOURCE = "expense_reimbursement_policy.md"
EXPECTED_SECTION = "Receipts"

EXPANDED_TERMS = [
    "itemized receipts",
    "individual expenses above €25",
    "receipt documentation",
    "proof of purchase",
    "expenses above 25 euros",
]


def main():
    documents = load_documents()

    if not documents:
        print("No document chunks found.")
        return

    expanded_query = (
        QUESTION + " " + " ".join(EXPANDED_TERMS)
    )

    queries = {
        "Original query": QUESTION,
        "Expanded query": expanded_query,
    }

    for query_name, query in queries.items():
        scores = bm25_scores(query, documents)

        ranked_results = sorted(
            zip(documents, scores),
            key=lambda item: item[1],
            reverse=True,
        )

        print(f"\n{'=' * 60}")
        print(query_name)
        print(f"{'=' * 60}")

        for rank, (document, score) in enumerate(
            ranked_results[:3],
            start=1,
        ):
            is_match = (
                document["source"] == EXPECTED_SOURCE
                and document["section_title"] == EXPECTED_SECTION
            )

            print(
                f"{rank}. {document['source']} | "
                f"{document['section_title']} | "
                f"score={score:.4f} | "
                f"match={is_match}"
            )


if __name__ == "__main__":
    main()
