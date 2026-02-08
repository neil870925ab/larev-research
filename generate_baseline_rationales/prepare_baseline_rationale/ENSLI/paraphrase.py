# Adapted from:
# https://github.com/HanjieChen/REV/blob/main/src/esnli_baseline/paraphrase.py

# https://huggingface.co/humarin/chatgpt_paraphraser_on_T5_base
# need new version of transformers
import csv
import os
import sys
import json
from tqdm import tqdm
import re
import time
import torch
import random
import numpy as np

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

def set_seed(seed):
    """
    Set the random seed for reproducibility
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def batch_indices(n_items, batch_size):
    for i in range(0, n_items, batch_size):
        yield list(range(i, min(i + batch_size, n_items)))

def paraphrase(question, tokenizer, model, device):
    max_length=200
    input_ids = tokenizer(
        f'paraphrase: {question}',
        return_tensors="pt", padding="longest",
        max_length=max_length,
        truncation=True,
    ).input_ids.to(device)
    
    outputs = model.generate(
        input_ids,
        max_length=max_length,
    )

    res = tokenizer.batch_decode(outputs, skip_special_tokens=True)

    return res

def main():

        csv.field_size_limit(sys.maxsize)

        
        task = 'ESNLI'
        SPLIT = ['train', 'val', 'test']

        current_path = os.path.dirname(os.path.abspath(__file__))
        tokenizer = AutoTokenizer.from_pretrained("humarin/chatgpt_paraphraser_on_T5_base")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        start_time = time.time()
        model = AutoModelForSeq2SeqLM.from_pretrained("humarin/chatgpt_paraphraser_on_T5_base")
        elapsed_time = time.time() - start_time
        print(f"Model loaded in {elapsed_time:.2f} seconds, using device: {device}")
        model.to(device)
        model.eval()
        batch_size = 32
        max_length = 200
        seed = 42
        set_seed(seed)


        for split in SPLIT:

                input_file = os.path.join(current_path, '../', '../', '../', 'dataset', task.lower(), 'output', task.lower()+'_'+split+'_templated.jsonl')
                input_file = os.path.normpath(input_file)
                output_file = os.path.join(current_path, '../', '../', 'output', task, 'baseline_rationales_'+split+'_output.jsonl')
                output_file = os.path.normpath(output_file)
                if not os.path.exists(input_file):
                        raise FileNotFoundError(f"Input file not found: {input_file}")
                os.makedirs(os.path.dirname(output_file), exist_ok=True)

                records = []

                with open(input_file, 'r', encoding="utf-8") as json_file:
                        for line in json_file:
                                line = line.strip()
                                if not line:
                                        continue
                                records.append(json.loads(line))

                rats = []
                for r in records:
                        rat = r["question_statement_text"]
                        rat = re.sub(r'[^\w\s]', '', rat)
                        rat = rat.strip() + '.'
                        rats.append(rat)

                print(f"[{split}] total samples: {len(records)}")

                gens = [None] * len(records)

                with torch.no_grad():
                        for idx_batch in tqdm(
                                batch_indices(len(records), batch_size), desc=f"[{split}] paraphrasing", total=(len(records)+batch_size-1)//batch_size):
                                batch_rats = [rats[i] for i in idx_batch]
                                batch_inputs = [f"paraphrase: {q}" for q in batch_rats]

                                enc = tokenizer(
                                        batch_inputs,
                                        return_tensors="pt",
                                        padding=True,
                                        truncation=True,
                                        max_length=max_length,
                                ).to(device)

                                out = model.generate(
                                        **enc,
                                        num_beams=1,
                                        max_length=max_length,
                                )
                                texts = tokenizer.batch_decode(out, skip_special_tokens=True)

                                for i, text in zip(idx_batch, texts):
                                        gens[i] = text

                with open(output_file, "w", encoding="utf-8") as f_out:
                        for rec, gen in zip(records, gens):
                                out_obj = {
                                        "question_text": rec["question_text"],
                                        "answer_text": rec["answer_text"],
                                        "question_statement_text": gen,
                                }
                                f_out.write(json.dumps(out_obj, ensure_ascii=False) + "\n")

                print(f"[{split}] Baseline rationales saved to -> {output_file}\n")

if __name__ == "__main__":
    main()
