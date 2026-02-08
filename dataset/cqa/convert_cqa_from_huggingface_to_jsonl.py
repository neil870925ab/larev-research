from datasets import load_dataset
import json
import os
import argparse

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--out_dir", required=True, type=str, help="Output directory for generated files.")
    args = parser.parse_args()

    dataset = load_dataset("commonsense_qa")
    combined = list(dataset["train"]) + list(dataset["validation"])

    os.makedirs(args.out_dir, exist_ok=True)
    output_file = os.path.join(args.out_dir, 'cqa.jsonl')

    with open(output_file, "w", encoding="utf-8") as f:
        for example in combined:
            # HuggingFace structure: choices is a dict {label:[], text:[]}
            choices = []
            if isinstance(example["choices"], dict):
                labels = example["choices"]["label"]
                texts = example["choices"]["text"]
                for label, text in zip(labels, texts):
                    choices.append({"label": label, "text": text})
            else:
                # Just in case, use the original structure
                choices = example["choices"]

            data = {
                "id": example["id"],
                "question": {
                    "stem": example["question"],
                    "choices": choices,
                    "question_concept": example.get("question_concept", "")
                },
                "answerKey": example["answerKey"]
            }
            f.write(json.dumps(data) + "\n")

    print(f"Output: {output_file}")

if __name__ == "__main__":
    main()
