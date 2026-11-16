
import json
from pathlib import Path

from src.evaluate_bm25 import load_documents
from src.evaluate_generalization import evaluate, print_results


# ==================================================
# Configuration
# ==================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

QUESTIONS_FILE = (
    PROJECT_ROOT
    / "data"
    / "evaluation"
    / "final_test_questions.json"
)


# ==================================================
# Load Questions
# ==================================================

def load_final_questions():

    with open(
        QUESTIONS_FILE,
        "r",
        encoding="utf-8",
    ) as file:

        questions = json.load(file)

    required_fields = {
        "question",
        "expected_source",
        "expected_section",
    }

    for index, item in enumerate(questions, start=1):

        missing_fields = required_fields - item.keys()

        if missing_fields:
            raise ValueError(
                f"Question {index} is missing fields: "
                f"{missing_fields}"
            )

    return questions


# ==================================================
# Main Evaluation
# ==================================================

def main():

    questions = load_final_questions()
    documents = load_documents()

    print("=" * 65)
    print("FINAL TEST EVALUATION")
    print("=" * 65)

    print(f"Document chunks: {len(documents)}")
    print(f"Test questions: {len(questions)}")

    methods = [
        "bm25",
        "hybrid",
        "query_expansion",
    ]

    all_results = []

    for method in methods:

        result = evaluate(
            questions=questions,
            documents=documents,
            method=method,
        )

        all_results.append(result)

        print_results(result)

    # --------------------------------------------------
    # Comparison
    # --------------------------------------------------

    print("\n" + "=" * 70)
    print("FINAL TEST COMPARISON")
    print("=" * 70)

    print(
        f"{'Method':<20}"
        f"{'Hit@1':<12}"
        f"{'Hit@3':<12}"
        f"{'MRR@3':<12}"
    )

    print("-" * 56)

    for result in all_results:

        print(
            f"{result['method']:<20}"
            f"{result['hit_rate_at_1']:<12.2%}"
            f"{result['hit_rate_at_3']:<12.2%}"
            f"{result['mrr_at_3']:<12.4f}"
        )

    print("\nEvaluation completed.")


if __name__ == "__main__":
    main()

