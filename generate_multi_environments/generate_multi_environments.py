"""
Generate multiple baseline rationale environments (E1–E3)
for IRM training in LAREV setting.

E1: Original baseline
E2: Mask baseline
E3: Antonym baseline
"""

import re
import json
import argparse
import os
import time
from pathlib import Path
from tqdm import tqdm

from google import genai
from google.genai import types

def _default_safety_settings():
    return [
        {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
    ]


def generate_antonym(sample, client, model_name, max_retries=3):
    question = sample.get("question_text", "")
    answer = sample.get("answer_text", "")
    text = sample.get("question_statement_text_masked", "")
    leakage_terms = sample.get("most_leaky_term", [])
    sample_id = sample.get("sample_id", sample.get("id", "unknown"))
    
    prompt = (
        "Please change the masked part of the following sentence with an antonym. \n"
        "Do not change any other part of the sentence except the masked part. \n"
        "Output only a single sentence. \n"
        f"Question context: {question}\n"
        f"Original answer: {answer}\n"
        f"Masked sentence: {text}\n"
        f"the masked term to be changed: {leakage_terms}\n"
        "Antonym version:"
    )

    config = types.GenerateContentConfig(
        temperature=0.0,
        max_output_tokens=256,
        safety_settings=_default_safety_settings(),
    )
    
    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=prompt,
                config=config,
            )

            out = (getattr(response, "text", None) or "").strip()
            if out:
                return out

            print(f"[EMPTY] Sample ID {sample_id}: empty response, using fallback.")
            return text.strip()

        except Exception as e:
            if attempt == max_retries - 1:
                print(f"[FAILED] Sample ID {sample_id}: failed after {max_retries} attempts - {e}")
                print("Using original text as fallback. Please regenerate later if needed.")
                return text.strip()
            else:
                print(f"[RETRY] Sample ID {sample_id} attempt {attempt + 1} failed: {e}, retrying...")
                time.sleep(2 ** attempt)

    return text.strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, required=True, choices=["ECQA", "ESNLI"], 
                        help="Task name (ECQA or ESNLI)")
    parser.add_argument("--model_name_or_path", type=str, required=True, choices=["t5-large", "bart-large"], 
                        help="What backbone model to use for computing IG scores (t5-large or bart-large)")
    parser.add_argument("--split_type", type=str, required=True, choices=["train", "val", "test"], 
                        help="Split type (train, val, test")
    parser.add_argument("--output", type=str, required=True, help="Path to output JSONL file")
    parser.add_argument("--gemini_key", type=str, required=True, help="Gemini API key")
    parser.add_argument("--model_name", type=str, default="gemini-2.5-flash", help="Gemini model name")
    args = parser.parse_args()

    client = genai.Client(api_key=args.gemini_key)

    current_path = os.path.dirname(os.path.abspath(__file__))
    
    input_file = os.path.join(current_path, '../', 'generate_baseline_rationales', 'output', 
                                  args.task, f'baseline_rationales_{args.split_type}_output_ig_{args.model_name_or_path}.jsonl')
    
    input_file = os.path.normpath(input_file)
    
    if not os.path.exists(input_file):
        print(f"Error: Input file '{input_file}' does not exist!")
        return
    
    print(f"Using input file: {input_file}")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load data
    data = []
    with open(input_file, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            sample = json.loads(line)
            # Add sample ID if not present
            if 'sample_id' not in sample and 'id' not in sample:
                sample['sample_id'] = idx+1
            data.append(sample)
    
    print(f"Total samples to process: {len(data)}")

    # Process data sequentially
    processed_count = 0
    with open(output_path, "w", encoding="utf-8") as out_f:
        for sample in tqdm(data, desc="Processing samples"):
            q = sample.get("question_text", "")
            a = sample.get("answer_text", "")
            b = sample.get("question_statement_text", "")
            c = sample.get("question_statement_text_masked", "")

            # E1: Original baseline
            E1 = b.strip()
            
            # E2: Mask baseline
            E2 = c.strip()
            
            # E3: Antonym baseline
            E3 = generate_antonym(sample, client, args.model_name)

            new_sample = {
                "question_text": q,
                "answer_text": a,
                "baseline": E1,
                "masked": E2,
                "antonym": E3,
            }

            out_f.write(json.dumps(new_sample, ensure_ascii=False) + "\n")
            processed_count += 1

    print(f"\nProcessing completed!")
    print(f"Output saved to: {output_path}")
    print(f"Processed {processed_count} samples")


if __name__ == "__main__":
    main()
