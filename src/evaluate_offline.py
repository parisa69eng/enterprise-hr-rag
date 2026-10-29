import json
import re
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent

SAMPLE_FILE = (
    PROJECT_ROOT
    / "data"
    / "evaluation"
    / "sample_answers.json"
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
    text = text.lower()

    for word, digit in NUMBER_WORDS.items():
        text = re.sub(rf"\b{word}\b", digit, text)

    text = text.replace("€", " euro ")
    text = re.sub(r"[^\w\s]", " ", text)

    return " ".join(text.split())


def fact_matches(answer: str, expected_fact) -> bool:
    if isinstance(expected_fact, str):
        alternatives = [expected_fact]
    else:
        alternatives = expected_fact

    normalized_answer = normalize(answer)

    return any(
        normalize(phrase) in normalized_answer
        for phrase in alternatives
    )


def evaluate_offline():
    if not SAMPLE_FILE.exists():
        raise FileNotFoundError(
            f"Sample file not found: {SAMPLE_FILE}"
        )

    samples = json.loads(
        SAMPLE_FILE.read_text(encoding="utf-8")
    )

    total_facts = 0
    matched_facts = 0
    successful_questions = 0

    print("Offline Answer Evaluation")
    print("=" * 60)

    for index, item in enumerate(samples, start=1):
        question = item["question"]
        answer = item["answer"]
        expected_facts = item["expected_facts"]

        print(f"\nQuestion {index}: {question}")
        print(f"Sample answer: {answer}")

        question_success = True

        for expected_fact in expected_facts:
            total_facts += 1
            matched = fact_matches(answer, expected_fact)

            if matched:
                matched_facts += 1
                status = "MATCH"
            else:
                question_success = False
                status = "MISSING"

            if isinstance(expected_fact, list):
                display_fact = " OR ".join(expected_fact)
            else:
                display_fact = expected_fact

            print(f"  [{status}] {display_fact}")

        if question_success:
            successful_questions += 1

    print("\n" + "=" * 60)
    print("Evaluation Summary")
    print(f"Sample answers: {len(samples)}")
    print(f"Matched facts: {matched_facts}/{total_facts}")

    if total_facts:
        print(
            f"Fact match rate: "
            f"{matched_facts / total_facts:.2%}"
        )

    print(
        f"Questions with all facts matched: "
        f"{successful_questions}/{len(samples)}"
    )


if __name__ == "__main__":
    evaluate_offline()
