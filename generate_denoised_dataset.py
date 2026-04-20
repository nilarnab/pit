#!/usr/bin/env python3

import argparse
import json
import os
import re
import time
import unicodedata
from pathlib import Path

import requests

from utils.defaults import EXTERNAL_LLM
from dotenv import load_dotenv
load_dotenv()

OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"

DENOISING_PROMPT = """\
You are a math reasoning expert. You will be given:
1. An original clean math question
2. The correct step-by-step solution to the original question
3. Multiple adversarial versions of the same question, each containing irrelevant noise and distractors

Your task: For EACH adversarial question, produce the correct step-by-step reasoning, \
identifying and ignoring all irrelevant information. Focus only on the mathematically \
relevant facts. The answer must be the same as the original for all of them.

Format your response EXACTLY like this — one block per adversarial question, separated by "---":
<reasoning steps for adversarial 1>
#### <ans>
---
<reasoning steps for adversarial 2>
#### <ans>
---
... and so on for each adversarial question.

The correct final answer is: {original_answer}

--- Original Question ---
{original_question}

--- Correct Solution ---
{original_raw}

--- Adversarial Questions ---
{adversarial_questions}

Provide the correct reasoning for each adversarial question now:\
"""


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKD", text)
    text = text.encode("ascii", "ignore").decode("ascii")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def parse_raw_response(raw: str):
    """Split a raw response on #### into (reasoning, answer)."""
    if "####" in raw:
        parts = raw.split("####", 1)
        reasoning = parts[0].strip().replace("\n", " ")
        answer = parts[1].strip().split()[0] if parts[1].strip() else ""
    else:
        reasoning = raw.strip().replace("\n", " ")
        answer = ""
    return reasoning, answer


def parse_batched_response(llm_raw: str, n: int) -> list:
    """Split batched LLM response into n individual reasoning blocks."""
    # Split on "---" separator lines
    blocks = re.split(r"\n---\n|^---$", llm_raw, flags=re.MULTILINE)
    # Filter empty blocks
    blocks = [b.strip() for b in blocks if b.strip()]
    if len(blocks) < n:
        # Fallback: try splitting on numbered headers if separator failed
        blocks_alt = re.split(r"\n(?=\*\*Adversarial \d+|\d+\.)", llm_raw)
        blocks_alt = [b.strip() for b in blocks_alt if b.strip()]
        if len(blocks_alt) >= n:
            blocks = blocks_alt
    return blocks


def normalize_answer(ans: str) -> str:
    """Strip currency, commas, whitespace; normalise to bare number string."""
    ans = ans.strip().replace("$", "").replace(",", "").replace("%", "")
    ans = ans.strip()
    try:
        f = float(ans)
        if f == int(f):
            return str(int(f))
        return str(f)
    except ValueError:
        return ans.lower()


def build_combined(reasoning: str, answer: str) -> str:
    reasoning = normalize_text(reasoning)
    answer = normalize_text(answer)
    return f"<reasoning>{reasoning}</reasoning><answer>{answer}</answer>"


def call_llm(
    prompt: str,
    api_key: str,
    model: str,
    delay: float,
    max_retries: int = 6,
    base_backoff: float = 5.0,
) -> str:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
    }
    for attempt in range(max_retries):
        response = requests.post(
            OPENROUTER_API_URL, headers=headers, json=payload, timeout=120
        )
        if response.status_code == 429:
            retry_after = response.headers.get("Retry-After")
            wait = float(retry_after) if retry_after else base_backoff * (2 ** attempt)
            print(f"  [429] Rate limited. Retrying in {wait:.1f}s (attempt {attempt+1}/{max_retries})")
            time.sleep(wait)
            continue
        response.raise_for_status()
        content = response.json()["choices"][0]["message"]["content"]
        time.sleep(delay)
        return content
    raise RuntimeError(f"Exceeded {max_retries} retries due to rate limiting.")


def make_clean_sample(entry: dict) -> dict:
    original_question = entry["original_question"]
    original_answer = entry["original_answer"]
    original_raw = entry["original_raw"]

    reasoning, answer = parse_raw_response(original_raw)
    # Prepend noise-acknowledgement note; no LLM call needed
    clean_reasoning = f"I did not find any noise or distractors in this question. {reasoning}"
    clean_raw = f"{clean_reasoning}\n#### {answer or original_answer}"
    combined = build_combined(clean_reasoning, answer or original_answer)

    return {
        "question": original_question,
        "original_question": original_question,
        "answer": original_answer,
        "raw": clean_raw,
        "reasoning": clean_reasoning,
        "combined": combined,
        "type": "clean",
        "answer_match": True,
    }


def make_adversarial_samples_batch(
    entry: dict,
    adversarial_questions: list,
    api_key: str,
    model: str,
    delay: float,
) -> list:
    """Call LLM once for all adversarial questions; return list of sample dicts."""
    original_question = entry["original_question"]
    original_answer = entry["original_answer"]
    original_raw = entry["original_raw"]

    numbered = "\n\n".join(
        f"[{i+1}] {q}" for i, q in enumerate(adversarial_questions)
    )

    prompt = DENOISING_PROMPT.format(
        original_answer=original_answer,
        original_question=original_question,
        original_raw=original_raw,
        adversarial_questions=numbered,
    )

    llm_raw = call_llm(prompt, api_key, model, delay)
    blocks = parse_batched_response(llm_raw, len(adversarial_questions))

    samples = []
    for i, adv_q in enumerate(adversarial_questions):
        block = blocks[i] if i < len(blocks) else ""
        reasoning, extracted_answer = parse_raw_response(block)
        answer_match = normalize_answer(extracted_answer) == normalize_answer(original_answer)
        combined = build_combined(reasoning, original_answer)
        clean_raw = f"{reasoning}\n#### {original_answer}"

        samples.append({
            "question": adv_q,
            "original_question": original_question,
            "answer": original_answer,
            "raw": clean_raw,
            "reasoning": reasoning,
            "combined": combined,
            "type": "adversarial",
            "answer_match": answer_match,
            "llm_raw_response": llm_raw,
            "block_index": i,
        })

    return samples


def load_existing_questions(output_path: Path) -> set:
    existing = set()
    if output_path.exists():
        with open(output_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        sample = json.loads(line)
                        existing.add(sample.get("question", ""))
                    except json.JSONDecodeError:
                        pass
    return existing


def generate(args):
    api_key = args.api_key or os.environ.get("OPENROUTER_API_KEY", "")
    if not api_key:
        raise ValueError(
            "No API key provided. Set OPENROUTER_API_KEY env var or pass --api-key."
        )

    model = args.model or EXTERNAL_LLM

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    existing_questions = load_existing_questions(output_path)
    if existing_questions:
        print(f"Found {len(existing_questions)} already-processed questions in {output_path}, skipping them.")

    entries = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))

    start = args.start_from
    end = args.end_to if args.end_to is not None else len(entries)
    entries = entries[start:end]

    total_clean = 0
    total_adversarial = 0
    total_match = 0
    total_mismatch = 0
    total_skipped = 0

    print(f"Processing entries [{start}:{end}] ({len(entries)} entries) from {input_path}")
    print(f"Model: {model}  |  delay: {args.delay}s\n")

    with open(output_path, "a", encoding="utf-8") as f_out:
        for idx, entry in enumerate(entries, start=start + 1):
            adversarials = entry["modified_questions"]["adverserials"]

            # --- Clean sample (no LLM) ---
            clean_q = entry["original_question"]
            if clean_q in existing_questions:
                total_skipped += 1
            else:
                clean_sample = make_clean_sample(entry)
                f_out.write(json.dumps(clean_sample) + "\n")
                f_out.flush()
                existing_questions.add(clean_q)
                total_clean += 1

            # --- Filter adversarials not yet processed ---
            pending_advs = [q for q in adversarials if q not in existing_questions]
            total_skipped += len(adversarials) - len(pending_advs)

            if pending_advs:
                try:
                    adv_samples = make_adversarial_samples_batch(
                        entry, pending_advs, api_key, model, args.delay
                    )
                    for adv_sample in adv_samples:
                        f_out.write(json.dumps(adv_sample) + "\n")
                        existing_questions.add(adv_sample["question"])
                        total_adversarial += 1
                        if adv_sample["answer_match"]:
                            total_match += 1
                        else:
                            total_mismatch += 1
                            print(
                                f"  [MISMATCH] entry {idx} block {adv_sample['block_index']} | "
                                f"expected={entry['original_answer']} | "
                                f"llm_raw={adv_sample['llm_raw_response'][:120]!r}"
                            )
                    f_out.flush()
                except Exception as exc:
                    print(f"  [ERROR] entry {idx}, all adversarials skipped: {exc}")

            total = total_clean + total_adversarial
            clean_pct = 100.0 * total_clean / total if total else 0
            print(
                f"[{idx}/{end}] samples={total}  skipped={total_skipped}  clean={clean_pct:.1f}%  "
                f"adv_match={total_match}/{total_adversarial}"
            )

    total = total_clean + total_adversarial
    print(f"\nDone. Written to {output_path}")
    print(f"Total new samples : {total}")
    print(f"Skipped (existing): {total_skipped}")
    if total:
        print(f"Clean             : {total_clean}  ({100*total_clean/total:.1f}%)")
        print(f"Adversarial       : {total_adversarial}  ({100*total_adversarial/total:.1f}%)")
    if total_adversarial:
        print(
            f"Answer match      : {total_match}/{total_adversarial}  "
            f"({100*total_match/total_adversarial:.1f}%)"
        )


def main():
    parser = argparse.ArgumentParser(
        description="Generate denoised training data using an external LLM."
    )
    parser.add_argument("--input", default="test.jsonl")
    parser.add_argument("--output", default="dataset/denoised_training.jsonl")
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--model", default=None,
                        help=f"Model to use (default: {EXTERNAL_LLM})")
    parser.add_argument("--start-from", type=int, default=0,
                        help="Start index (0-based, inclusive)")
    parser.add_argument("--end-to", type=int, default=None,
                        help="End index (0-based, exclusive, default: all)")
    parser.add_argument("--delay", type=float, default=10,
                        help="Seconds to wait after each API call (default: 10)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    generate(args)


if __name__ == "__main__":
    main()
