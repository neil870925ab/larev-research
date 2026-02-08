import pandas as pd
import torch
import time
import os
import re
import argparse
import random
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

def set_seed(args):
    """
    Set the random seed for reproducibility
    """
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def normalize_spaces(s: str) -> str:
    s = s.replace('\n', ' ')
    s = re.sub(r'\s{2,}', ' ', s) # Replace multiple spaces with a single space
    return s.strip()
    

def batch(iterable, n=16):
    for i in range(0, len(iterable), n):
        yield iterable[i:i+n]

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for initialization."
    )
    args = parser.parse_args()

    set_seed(args)

    current_path = os.path.dirname(os.path.abspath(__file__))

    model_dir = os.path.join(current_path, 'model_data', 'question-converter-t5-3b')
    model_dir = os.path.normpath(model_dir)
    if not os.path.exists(model_dir):
        raise FileNotFoundError(f"Model directory not found: {model_dir}")


    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    start_time = time.time()
    model = AutoModelForSeq2SeqLM.from_pretrained(model_dir, local_files_only=True).to(device)
    elapsed_time = time.time() - start_time
    print(f"Model loaded in {elapsed_time:.2f} seconds, using device: {device}")
    model.eval()

    task = 'ECQA'
    SPLIT = ["train", "val", "test"]

    for split in SPLIT:

        input_file = os.path.join(current_path, '../', '../', '../', '../', 'dataset', task.lower(), 'output', task.lower()+'_'+split+'.csv')
        # ECQA dataset created by generate_data.py; The output is located at path/to/larev-research/dataset/ecqa/output/
        input_file = os.path.normpath(input_file)
        output_file = os.path.join(current_path, '../', '../', '../', 'output', task, 'baseline_rationales_'+split+'_output.jsonl')
        output_file = os.path.normpath(output_file)
        if not os.path.exists(input_file):
                raise FileNotFoundError(f"Input file not found: {input_file}")
        os.makedirs(os.path.dirname(output_file), exist_ok=True)

        df = pd.read_csv(input_file)
        questions = df["q_text"].astype(str).apply(normalize_spaces).tolist()
        answers = df["q_ans"].astype(str).apply(normalize_spaces).tolist()


        inputs = [f"{q} </s> {a}" for q, a in zip(questions, answers)]
        pairs = list(zip(inputs, answers))
        gens = []

        with torch.no_grad():
            for chunk in tqdm(list(batch(list(enumerate(pairs)), 16)),
                            total=(len(pairs) + 15)//16,
                            desc=f"Generating baseline rationales for ECQA)"):

                # chunk: list of (global_idx, (input_text, answer))
                chunk_inputs  = [item[1][0] for item in chunk]

                enc = tokenizer(
                    chunk_inputs,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=128
                ).to(device)

                out = model.generate(**enc, num_beams=1, max_length=128)
                texts = tokenizer.batch_decode(out, skip_special_tokens=True)
                gens.extend(texts)


        print(f"Total samples: {len(df)}")

        out_df = pd.DataFrame({
            "question_text": questions,
            "answer_text": answers,
            "question_statement_text": gens
        })
        out_df.to_json(output_file, lines=True, orient="records", force_ascii=False)
        print(f"ECQA baseline rationales saved to -> {output_file}")

if __name__ == "__main__":
    main()


