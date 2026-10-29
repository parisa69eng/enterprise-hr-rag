import json
import re
from pathlib import Path

from src.rag import generate_answer


PROJECT_ROOT = Path(__file__).resolve().parent.parent

EVALUATION_FILE = (
    PROJECT_ROOT
    / "data"
    / "evaluation"
    / "answer_quality_questions.json"
)


NUMBER_WORDS = {
    "zero": "0",
    "one": "1",
    "two": "2",
    "three": "3",
    "four": "4",
    "five": "5",
    "six": "6",
    "seven": "7",
    "eight": "8",
    "nine": "9",
    "ten": "10",
}


def normalize(text: str) -> str:
    """Normalize text for basic fact matching."""
    text = text.lower()

    for word, digit in NUMBER_WORDS.items():
        text = re.sub(rf"\b{word}\b", digit, text)

    text = text.replace("€", " euro ")
    text = re.sub(r"[^\w\s]", " ", text)

    return " ".join(text.split())


def fact_matches(answer: str, expected_fact) -> bool:
    """Match a fact using one or more accepted phrasings."""
    if isinstance(expected_fact, str):
        alternatives = [expected_fact]
    else:
        alternatives = expected_fact

    normalized_answer = normalize(answer)

    return any(
        normalize(phrase) in normalized_answer
        for phrase in alternatives
    )


def evaluate_answer_quality() -> None:
    if not EVALUATION_FILE.exists():
        raise FileNotFoundError(
            f"Evaluation file not found: {EVALUATION_FILE}"
        )

    questions = json.loads(
        EVALUATION_FILE.read_text(encoding="utf-8")
    )

    if not isinstance(questions, list):
        raise ValueError(
            "The evaluation file must contain a JSON list."
        )

    total_facts = 0
    matched_facts = 0
    evaluated_questions = 0
    questions_with_all_facts = 0
    failed_questions = 0
    api_errors = 0

    for index, item in enumerate(questions, start=1):
        question = item["question"]
        expected_facts = item["expected_facts"]

        print("\n" + "=" * 70)
        print(f"Question {index}: {question}")

        try:
            answer = generate_answer(question)
        except Exception as error:
            api_errors += 1
            print(f"  [API ERROR] {type(error).__name__}: {error}")
            print("  This question was not evaluated.")
            continue

        evaluated_questions += 1

        print(f"\nAnswer:\n{answer}")
        print("\nFact checks:")

        question_matches = 0

        for expected_fact in expected_facts:
            total_facts += 1

            is_match = fact_matches(answer, expected_fact)

            if is_match:
                matched_facts += 1
                question_matches += 1
                status = "MATCH"
            else:
                status = "MISSING"

            if isinstance(expected_fact, list):
                display_fact = " OR ".join(expected_fact)
            else:
                display_fact = expected_fact

            print(f"  [{status}] {display_fact}")

        all_facts_matched = (
            question_matches == len(expected_facts)
        )

        if all_facts_matched:
            questions_with_all_facts += 1
        else:
            failed_questions += 1

    print("\n========== Evaluation Summary ==========")
    print(f"Total questions: {len(questions)}")
    print(f"Evaluated questions: {evaluated_questions}")
    print(f"Questions with API errors: {api_errors}")
    print(f"Questions missing expected facts: {failed_questions}")
    print(f"Matched facts: {matched_facts}/{total_facts}")

    if total_facts > 0:
        print(
            f"Fact match rate: "
            f"{matched_facts / total_facts:.2%}"
        )
    else:
        print("Fact match rate: N/A (no facts evaluated)")

    print(
        "Questions with all expected facts: "
        f"{questions_with_all_facts}/{evaluated_questions}"
    )

    if evaluated_questions > 0:
        print(
            "Complete-fact coverage rate: "
            f"{questions_with_all_facts / evaluated_questions:.2%}"
        )
    else:
        print("Complete-fact coverage rate: N/A")

    if api_errors:
        print(
            "\nWARNING: Evaluation is incomplete because "
            "some questions could not be evaluated."
        )


if __name__ == "__main__":
    evaluate_answer_quality()
