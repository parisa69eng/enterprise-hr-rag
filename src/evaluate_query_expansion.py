from pathlib import Path
import json

from src.evaluate_bm25 import (
    bm25_scores,
    load_documents,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent

QUESTIONS_PATH = (
    PROJECT_ROOT
    / "data"
    / "evaluation"
    / "retrieval_questions.json"
)


QUERY_EXPANSIONS = {
    "What documentation is needed for an extended sickness absence?": [
        "medical certificate",
        "illness",
        "absence",
        "sick leave",
        "working days",
    ],

    "What conditions must be met before an employee can work remotely?": [
        "probation period",
        "completed probation",
        "manager approval",
        "eligibility requirements",
        "remote work request",
    ],

    "What security responsibilities do remote workers have?": [
        "secure working environment",
        "protecting company information",
        "information security",
        "data protection",
        "security measures",
        "employee responsibilities",
    ],

    "How should an employee apply for remote work?": [
        "submit a remote work request",
        "submitted through the HR portal",
        "HR portal",
        "manager must approve",
        "manager approval",
        "request before arrangement begins",
    ],

    "What proof must I provide for an expensive business purchase?": [
        "itemized receipts",
        "individual expenses above €25",
        "receipt documentation",
        "proof of purchase",
        "expenses above 25 euros",
    ],
}


def load_questions():
    with open(QUESTIONS_PATH, "r", encoding="utf-8") as file:
        return json.load(file)


def evaluate(documents, questions, use_expansion=False):
    hits_at_1 = 0
    hits_at_3 = 0
    reciprocal_rank_sum = 0

    failures = []

    for item in questions:
        question = item["question"]

        expected_source = item["expected_source"]
        expected_section = item["expected_section"]

        query = question

        if use_expansion and question in QUERY_EXPANSIONS:
            extra_terms = QUERY_EXPANSIONS[question]
            query = query + " " + " ".join(extra_terms)

        scores = bm25_scores(query, documents)

        ranked_results = sorted(
            zip(documents, scores),
            key=lambda item: item[1],
            reverse=True,
        )

        top_3 = ranked_results[:3]

        correct_rank = None

        for rank, (document, score) in enumerate(
            top_3,
            start=1,
        ):
            if (
                document["source"] == expected_source
                and document["section_title"] == expected_section
            ):
                correct_rank = rank
                break

        if correct_rank == 1:
            hits_at_1 += 1

        if correct_rank is not None:
            hits_at_3 += 1
            reciprocal_rank_sum += 1 / correct_rank
        else:
            failures.append(question)

    total = len(questions)

    return {
        "hit_rate_at_1": hits_at_1 / total,
        "hit_rate_at_3": hits_at_3 / total,
        "mrr_at_3": reciprocal_rank_sum / total,
        "failures": failures,
    }


def print_results(title, results):
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)

    print(
        f"Hit Rate@1: "
        f"{results['hit_rate_at_1']:.2%}"
    )

    print(
        f"Hit Rate@3: "
        f"{results['hit_rate_at_3']:.2%}"
    )

    print(
        f"MRR@3: "
        f"{results['mrr_at_3']:.4f}"
    )

    print("\nFailed questions:")

    if results["failures"]:
        for question in results["failures"]:
            print(f"- {question}")
    else:
        print("None")


def main():
    documents = load_documents()
    questions = load_questions()

    print(f"Total documents: {len(documents)}")
    print(f"Total questions: {len(questions)}")
    print(
        f"Questions with expansion: "
        f"{len(QUERY_EXPANSIONS)}"
    )

    baseline_results = evaluate(
        documents,
        questions,
        use_expansion=False,
    )

    expanded_results = evaluate(
        documents,
        questions,
        use_expansion=True,
    )

    print_results(
        "BASELINE BM25",
        baseline_results,
    )

    print_results(
        "BM25 + QUERY EXPANSION",
        expanded_results,
    )


if __name__ == "__main__":
    main()
