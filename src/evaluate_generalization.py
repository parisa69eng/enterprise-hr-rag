
import json
from pathlib import Path

from src.evaluate_bm25 import (
    bm25_scores,
    load_documents,
    tokenize,
)


# ==================================================
# Configuration
# ==================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

QUESTIONS_FILE = (
    PROJECT_ROOT
    / "data"
    / "evaluation"
    / "generalization_questions.json"
)

TITLE_BOOST = 0.5
TOP_K = 3


# ==================================================
# Intent-based Query Expansion Rules
# ==================================================

INTENT_RULES = [
    {
        "intent": "leave_request",
        "triggers": [
            "vacation request",
            "holiday request",
            "place a vacation",
            "annual leave request",
        ],
        "expansions": [
            "annual leave",
            "HR portal",
            "manager review",
            "five working days",
        ],
    },
    {
        "intent": "leave_approval",
        "triggers": [
            "responsible for considering",
            "who considers",
            "who reviews",
            "who is responsible",
        ],
        "expansions": [
            "annual leave",
            "manager review",
            "manager approval",
        ],
    },
    {
        "intent": "sick_leave",
        "triggers": [
            "doctor's note",
            "medical note",
            "sickness",
            "illness",
            "medical certificate",
        ],
        "expansions": [
            "medical certificate",
            "sick leave",
            "three consecutive working days",
        ],
    },
    {
        "intent": "remote_request",
        "triggers": [
            "work from home request",
            "apply for remote work",
            "request to work remotely",
            "submit a request to work from home",
            "remote work request",
        ],
        "expansions": [
            "HR portal",
            "manager approval",
            "request before arrangement begins",
        ],
    },
    {
        "intent": "remote_eligibility",
        "triggers": [
            "probation",
            "eligibility",
            "conditions for remote work",
            "can someone work remotely",
        ],
        "expansions": [
            "completed probation period",
            "manager approval",
        ],
    },
    {
        "intent": "remote_equipment",
        "triggers": [
            "computer",
            "laptop",
            "supplies the computer",
            "who supplies",
            "equipment",
        ],
        "expansions": [
            "company provides an approved laptop",
            "remote work equipment",
        ],
    },
    {
        "intent": "remote_security",
        "triggers": [
            "confidential",
            "security responsibilities",
            "protect company data",
            "secure working environment",
            "information security",
        ],
        "expansions": [
            "secure working environment",
            "protecting company information",
            "data protection",
        ],
    },
    {
        "intent": "remote_schedule",
        "triggers": [
            "days per week",
            "office days",
            "remote work schedule",
            "capped or unlimited",
        ],
        "expansions": [
            "up to three days per week",
            "office days",
        ],
    },
    {
        "intent": "expense_deadline",
        "triggers": [
            "how long after incurring",
            "reimbursement claim be made",
            "submission deadline",
            "when can a claim",
        ],
        "expansions": [
            "submit expense claims",
            "within 30 calendar days",
            "expense reimbursement",
        ],
    },
    {
        "intent": "expense_approval",
        "triggers": [
            "doesn't comply",
            "does not comply",
            "claim doesn't comply",
            "claim does not comply",
            "reimbursement rules",
        ],
        "expansions": [
            "expense claim",
            "policy requirements",
            "manager approval",
            "may be rejected",
        ],
    },
    {
        "intent": "expense_receipts",
        "triggers": [
            "receipt",
            "proof of purchase",
            "detailed receipt",
            "itemized receipt",
        ],
        "expansions": [
            "itemized receipts",
            "individual expenses",
            "above €25",
        ],
    },
    {
        "intent": "expense_eligibility",
        "triggers": [
            "reimbursed",
            "private shopping",
            "eligible expenses",
            "work-related purchases",
        ],
        "expansions": [
            "approved business travel",
            "business meals",
            "work supplies",
            "personal expenses are not eligible",
        ],
    },
]


# ==================================================
# Intent Classification and Query Expansion
# ==================================================

def detect_intent(query: str):
    """
    Detect the most relevant intent using keyword and phrase rules.

    Returns:
        The matching rule or None.
    """

    normalized_query = query.lower()

    matched_rules = []

    for rule in INTENT_RULES:

        matched_triggers = [
            trigger
            for trigger in rule["triggers"]
            if trigger in normalized_query
        ]

        if matched_triggers:
            matched_rules.append(
                (
                    rule,
                    len(matched_triggers),
                    max(len(trigger) for trigger in matched_triggers),
                )
            )

    if not matched_rules:
        return None

    # Prefer the rule with more matched triggers.
    # If tied, prefer the longer and more specific phrase.
    matched_rules.sort(
        key=lambda item: (item[1], item[2]),
        reverse=True,
    )

    return matched_rules[0][0]


def expand_query(query: str):
    """
    Expand a query using only the detected intent.

    Returns:
        expanded_query, detected_intent
    """

    matched_rule = detect_intent(query)

    if matched_rule is None:
        return query, None

    expansions = matched_rule["expansions"]

    expanded_query = (
        query + " " + " ".join(expansions)
    )

    return expanded_query, matched_rule["intent"]


# ==================================================
# Load Evaluation Questions
# ==================================================

def load_questions() -> list[dict]:

    with open(
        QUESTIONS_FILE,
        "r",
        encoding="utf-8",
    ) as file:

        return json.load(file)


# ==================================================
# Title Scoring
# ==================================================

def calculate_title_score(
    query: str,
    title: str,
) -> float:

    query_tokens = set(tokenize(query))
    title_tokens = set(tokenize(title))

    if not title_tokens:
        return 0.0

    common_tokens = query_tokens.intersection(
        title_tokens
    )

    return len(common_tokens) / len(title_tokens)


# ==================================================
# Document Ranking
# ==================================================

def rank_documents(
    query: str,
    documents: list[dict],
    method: str = "bm25",
) -> list[tuple[dict, float]]:
    """
    Rank documents using BM25 or Hybrid Search.

    Returns:
        [(document, score), ...]
    """

    content_scores = bm25_scores(
        query,
        documents,
    )

    results = []

    for document, content_score in zip(
        documents,
        content_scores,
    ):

        final_score = content_score

        if method == "hybrid":

            title_score = calculate_title_score(
                query,
                document.get("section_title", ""),
            )

            final_score += (
                TITLE_BOOST * title_score
            )

        results.append(
            (document, final_score)
        )

    results.sort(
        key=lambda item: item[1],
        reverse=True,
    )

    return results


# ==================================================
# Evaluation
# ==================================================

def evaluate(
    questions: list[dict],
    documents: list[dict],
    method: str,
) -> dict:

    hits_at_1 = 0
    hits_at_3 = 0
    reciprocal_rank_sum = 0.0

    failures = []
    detected_intents = {}

    for item in questions:

        original_query = item["question"]

        expected_source = item["expected_source"]
        expected_section = item["expected_section"]

        query = original_query

        if method == "query_expansion":

            query, intent = expand_query(
                original_query
            )

            detected_intents[intent or "no_intent"] = (
                detected_intents.get(
                    intent or "no_intent",
                    0,
                ) + 1
            )

        ranking_method = (
            "hybrid"
            if method == "hybrid"
            else "bm25"
        )

        results = rank_documents(
            query=query,
            documents=documents,
            method=ranking_method,
        )

        top_results = results[:TOP_K]

        matching_ranks = []

        for rank, (document, score) in enumerate(
            top_results,
            start=1,
        ):

            is_match = (
                document["source"] == expected_source
                and document["section_title"]
                == expected_section
            )

            if is_match:
                matching_ranks.append(rank)

        if matching_ranks:

            best_rank = matching_ranks[0]

            if best_rank == 1:
                hits_at_1 += 1

            hits_at_3 += 1

            reciprocal_rank_sum += (
                1.0 / best_rank
            )

        else:

            failures.append({
                "question": original_query,
                "expected_source": expected_source,
                "expected_section": expected_section,
            })

    total_questions = len(questions)

    if total_questions == 0:
        raise ValueError(
            "No evaluation questions were found."
        )

    return {
        "method": method,
        "hit_rate_at_1": hits_at_1 / total_questions,
        "hit_rate_at_3": hits_at_3 / total_questions,
        "mrr_at_3": reciprocal_rank_sum / total_questions,
        "failures": failures,
        "detected_intents": detected_intents,
    }


# ==================================================
# Print Results
# ==================================================

def print_results(result: dict):

    print("\n" + "=" * 55)
    print(result["method"].upper())
    print("=" * 55)

    print(
        f"Hit Rate@1: "
        f"{result['hit_rate_at_1']:.2%}"
    )

    print(
        f"Hit Rate@3: "
        f"{result['hit_rate_at_3']:.2%}"
    )

    print(
        f"MRR@3: "
        f"{result['mrr_at_3']:.4f}"
    )

    print(
        f"Failed questions: "
        f"{len(result['failures'])}"
    )

    if result["method"] == "query_expansion":

        print("\nDetected intents:")

        for intent, count in result[
            "detected_intents"
        ].items():

            print(
                f"- {intent}: {count}"
            )

    if result["failures"]:

        print("\nFailures:")

        for failure in result["failures"]:

            print(
                f"- {failure['question']} "
                f"| Expected: "
                f"{failure['expected_section']}"
            )


# ==================================================
# Main
# ==================================================

def main():

    questions = load_questions()
    documents = load_documents()

    print("=" * 55)
    print("GENERALIZATION EVALUATION")
    print("=" * 55)

    print(
        f"Document chunks: {len(documents)}"
    )

    print(
        f"Test questions: {len(questions)}"
    )

    print(
        f"Hybrid title boost: {TITLE_BOOST}"
    )

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

    # Final comparison
    print("\n" + "=" * 65)
    print("FINAL COMPARISON")
    print("=" * 65)

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


if __name__ == "__main__":
    main()

