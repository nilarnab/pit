"""
1. Make a paraphrase maker
2. run sft

"""
import time
import re

from openai import OpenAI
from dotenv import load_dotenv
import os

from utils.defaults import EXTERNAL_LLM

load_dotenv()

from create_adverserial_dataset_test import ask_a_math_question

DEEP_SEEK_API_KEY = str(os.getenv("DEEP_SEEK_API_KEY"))

client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=DEEP_SEEK_API_KEY,
        )

def make_story_by_calling_genai(prompt: str, history):
    # print("PROMPT:", prompt)
    # print("HISTORY:", history)
    time_wait = 2
    for retry in range(20):
        try:
            completion = client.chat.completions.create(
                # extra_headers={
                #   "HTTP-Referer": "<YOUR_SITE_URL>", # Optional. Site URL for rankings on openrouter.ai.
                #   "X-Title": "<YOUR_SITE_NAME>", # Optional. Site title for rankings on openrouter.ai.
                # },
                model=EXTERNAL_LLM,
                messages=history
            )
            # print(f"Made completion: {completion}")
            story_raw = completion.choices[0].message.content
            if story_raw is None:
                raise Exception("Generated NONE question")


            return story_raw
        except Exception as error:
            print("Got error while making LLM call", str(error), "will wait for", time_wait)
            time.sleep(time_wait)
            time_wait = time_wait * 2
            continue

    raise Exception("EXTERNAL LLM SERVICE DOWN")
    return None


_ADVERSARIAL_SYSTEM = (
    "You are a dataset generator for LLM robustness research. "
    "Your job is to inject irrelevant distractor sentences into math word problems "
    "so that a language model might be misled into using the wrong numbers."
)

_ADVERSARIAL_USER = """\
ORIGINAL QUESTION: "{question}"

Task:
- Insert between 1 and 3 new distractor sentences anywhere in the question.
- Each distractor must be numerically rich (contain specific numbers/quantities), \
feel topically plausible, and be completely irrelevant to computing the correct answer.
- Do NOT change any of the original sentences.
- The question must remain answerable with the same correct answer.

Reply with ONLY the modified question — no explanation, no preamble.

Example:
ORIGINAL QUESTION: "Natalia sold clips to 48 of her friends in April, and then she \
sold half as many clips in May. How many clips did Natalia sell altogether in April and May?"

MODIFIED QUESTION: Natalia sold clips to 48 of her friends in April, and then she sold \
half as many clips in May. Her brother collected 37 stamps last Tuesday and traded 12 of \
them with a friend. How many clips did Natalia sell altogether in April and May?\
"""

_RETRY_TOO_COMPLEX = (
    "The model could not extract any numeric answer — the question became too confusing. "
    "Simplify the distractors (fewer numbers, shorter sentences) and try again. "
    "Reply with ONLY the modified question."
)

_RETRY_CORRECT = (
    "The model still got the right answer — the distractors were not misleading enough. "
    "Make them harder by using numbers that are close to or derived from the real values in the problem. "
    "Reply with ONLY the modified question."
)

_RETRY_FOOLED = (
    "The distractors successfully fooled the model. "
    "Now generate a fresh variant with different distractor sentences for dataset variety. "
    "Reply with ONLY the modified question."
)


def get_paraphrase(question):
    history = [
        {"role": "system", "content": _ADVERSARIAL_SYSTEM},
        {"role": "user", "content": _ADVERSARIAL_USER.format(question=question)},
    ]
    return make_story_by_calling_genai("", history)


def make_adverserials_for_one_question(question, answer_ref, limit=1, max_iteration_count=20):
    adverserials = []
    answers = []
    responses = []

    history = [
        {"role": "system", "content": _ADVERSARIAL_SYSTEM},
        {"role": "user", "content": _ADVERSARIAL_USER.format(question=question)},
    ]
    print("LIMIT:", limit)

    for _ in range(max_iteration_count):
        if len(adverserials) >= limit:
            break

        print("Adverserial count", len(adverserials))

        new_question = make_story_by_calling_genai("", history)
        print("Modified question:", new_question[:300])

        history.append({"role": "assistant", "content": new_question})

        response, answer = ask_a_math_question(new_question)
        print("QUESTION ASKED:", new_question[:200])
        print("response:", response)

        answer_stripped = answer.strip() if answer is not None else None

        if answer_stripped is not None:
            print("EXTRACTED ANSWER:", answer_stripped, "REFERENCE ANSWER:", answer_ref,
                  "MATCH:", float(answer_stripped) == float(answer_ref))
        else:
            print("Model returned no numeric answer.")

        try:
            if answer_stripped is None:
                adverserials.append(new_question)
                answers.append(answer_stripped)
                responses.append(response)
                history.append({"role": "user", "content": _RETRY_TOO_COMPLEX})
            elif float(answer_stripped) != float(answer_ref):
                # Model was fooled — good adversarial example
                adverserials.append(new_question)
                answers.append(answer_stripped)
                responses.append(response)
                history.append({"role": "user", "content": _RETRY_FOOLED})
            else:
                # Model answered correctly — distractors weren't misleading enough
                history.append({"role": "user", "content": _RETRY_CORRECT})
        except Exception as error:
            print("Error during answer comparison:", str(error))

    return {
        "adverserials": adverserials,
        "answers": answers,
        "responses": responses,
    }


def make_adverserial_questions(input_file_path, output_file_path=None, limit_per_question=1, start_from=1,end_at=None):
    """
    Reads a JSON array file of question/answer records, generates adversarial variants
    for each, and appends results to an output JSONL file.

    Args:
        input_file_path    : Path to input JSON file (a list of objects with at least
                             "question" and "answer" keys).
        output_file_path   : Path to output JSONL file. Defaults to
                             <input_stem>_adversarial.jsonl next to the input file.
        limit_per_question : Max successful adversarial variants to collect per question.
    """
    import json
    from pathlib import Path

    input_path = Path(input_file_path)
    if output_file_path is None:
        output_file_path = input_path.parent / f"{input_path.stem}_adversarial.jsonl"

    output_path = Path(output_file_path)

    with open(input_path, "r", encoding="utf-8") as infile:
        records = json.load(infile)  # parse the full JSON array

    with open(output_path, "a", encoding="utf-8") as outfile:
        for idx, record in enumerate(records[start_from:end_at], start=start_from + 1):
            question   = record.get("question")
            print("QUESTION", question)
            answer_ref = record.get("answer")

            if question is None or answer_ref is None:
                print(f"[Record {idx}] Skipping — missing 'question' or 'answer' key.")
                continue

            print(f"\n[Record {idx}/{len(records)}] Generating adversarials for: {question!r}")

            results = make_adverserials_for_one_question(
                question, answer_ref, limit=limit_per_question
            )

            print("results", results)

            out_record = {
                "original_question": question,
                "original_answer": answer_ref,
                "original_reasoning": record.get("reasoning"),
                "original_raw": record.get("raw"),
                "modified_questions": results,
            }
            outfile.write(json.dumps(out_record) + "\n")

            outfile.flush()  # persist after each question in case of crash
            print(f"[Record {idx}] Wrote {len(results)} adversarial record(s).")

    print(f"\nDone. Results appended to: {output_path}")


if __name__ == "__main__":
#     prompt = """QUESTION: "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. Natalia also sold 49 clips on June. How many clips did Natalia sell altogether in April and May?"
#
# Instructions: Add noise to the QUESTION such that an llm solving this will get confused. Add random numbers that are not relevant. Answer ONLY the modified question."""
#     completion = make_story_by_calling_genai(prompt)
#
#     print(completion)
#     question = "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. Natalia also sold 49 clips on June. How many clips did Natalia sell altogether in April and May?"
#     adv_ques = make_adverserials_for_one_question(question, 72, limit=5)
#
#     print("adv question", adv_ques)
    print("Starting to make quesyions")
    import argparse

    parser = argparse.ArgumentParser(description="Generate adversarial math questions.")
    parser.add_argument(
        "--input",
        type=str,
        default="dataset/gsm8k_processed_train.json",
        help="Path to input JSON file (default: dataset/gsm8k_processed_train.json)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to output JSONL file (default: <input_stem>_adversarial.jsonl)",
    )
    parser.add_argument(
        "--start-from",
        type=int,
        default=1,
        help="Index of the record to start from, 0-based (default: 1)",
    )
    parser.add_argument(
        "--limit-per-question",
        type=int,
        default=3,
        help="Min adversarial variants to collect per question (default: 1)",
    )
    parser.add_argument(
        "--end-at",
        type=int,
        default=None,
        help="Index of the record to stop at, exclusive, 0-based (default: None = process all)",
    )
    args = parser.parse_args()

    print("Starting to make questions")
    make_adverserial_questions(
        input_file_path=args.input,
        output_file_path=args.output,
        limit_per_question=args.limit_per_question,
        start_from=args.start_from,
        end_at=args.end_at,
    )
    # make_adverserial_questions("dataset/gsm8k_processed_train.json")


