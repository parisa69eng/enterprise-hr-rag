import json
from pathlib import Path

from src.rag import generate_answer


PROJECT_ROOT = Path(__file__).resolve().parent.parent
EVALUATION_FILE = (
    PROJECT_ROOT
    / "data"
    / "evaluation"
    / "answerability_questions.json"
)

ABSTENTION_MARKERS = (
    "could not find",
    "couldn't find",
    "does not mention",
    "does not specify",
    "not mentioned",
    "not covered",
    "not provide",
    "no information",
    "does not contain",
)


def looks_like_abstention(answer: str) -> bool:
    answer_lower = answer.lower()
    return any(
        marker in answer_lower
        for marker in ABSTENTION_MARKERS
    )


def evaluate_answers() -> None:
    questions = json.loads(
        EVALUATION_FILE.read_text(encoding="utf-8")
    )

    passed = 0
    total = len(questions)

    for index, item in enumerate(questions, start=1):
        question = item["question"]
        expected_answerable = item["expected_answerable"]

        print(f"\n{'=' * 70}")
        print(f"Question {index}: {question}")

        try:
            answer = generate_answer(question)
            detected_abstention = looks_like_abstention(answer)

            # For this first version, check whether the
            # response behavior matches the expected answerability.
            passed_test = (
                detected_abstention == (not expected_answerable)
            )

            if passed_test:
                passed += 1

            print(f"Answer:\n{answer}")
            print(f"Detected abstention: {detected_abstention}")
            print(f"Evaluation: {'PASS' if passed_test else 'FAIL'}")

        except Exception as error:
            print(f"Error: {error}")
            print("Evaluation: ERROR")

    print("\n========== Evaluation Summary ==========")
    print(f"Total questions: {total}")
    print(f"Passed: {passed}")
    print(f"Failed: {total - passed}")

    if total:
        print(f"Pass rate: {passed / total:.2%}")


if __name__ == "__main__":
    evaluate_answers()
