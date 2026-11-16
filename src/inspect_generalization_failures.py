from src.evaluate_generalization import (
    load_questions,
    load_documents,
    rank_documents,
    expand_query,
    TOP_K,
)


def main():
    questions = load_questions()
    documents = load_documents()

    target_question = (
        "Who supplies the computer used for working remotely?"
    )

    for item in questions:
        if item["question"] != target_question:
            continue

        print("\nQUESTION:", target_question)
        print(
            "EXPECTED:",
            item["expected_source"],
            "|",
            item["expected_section"],
        )

        expanded_query = expand_query(target_question)

        print("\nEXPANDED QUERY:")
        print(expanded_query)

        for method in ["bm25", "hybrid", "query_expansion"]:

            query = (
                expanded_query
                if method == "query_expansion"
                else target_question
            )

            ranking_method = (
                "hybrid" if method == "hybrid" else "bm25"
            )

            results = rank_documents(
                query,
                documents,
                ranking_method,
            )

            print(f"\n{method.upper()}:")

            for rank, (document, score) in enumerate(
                results[:TOP_K],
                start=1,
            ):
                match = (
                    document["source"] == item["expected_source"]
                    and document["section_title"]
                    == item["expected_section"]
                )

                print(
                    f"{rank}. {document['section_title']} "
                    f"| {score:.2f} | match={match}"
                )

        break


if __name__ == "__main__":
    main()
