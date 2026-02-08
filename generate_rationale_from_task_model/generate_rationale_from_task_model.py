import os
import json
import argparse
import torch
import pandas as pd
import random
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM

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

def build_task_prompt(task, ex):
    if task == "ECQA":
        prompt = f"""Question: {ex['question_text']}
Answer: {ex['answer_text']}

Explain why the answer is correct in detail.
Assume the given answer is correct. Do not challenge or change the answer.
Please finish in one sentence.
"""
    elif task == "ESNLI":
        prompt = f"""Premise: {ex['premise']}
Hypothesis: {ex['hypothesis']}
Answer: {ex['answer_text']}

Explain why the answer is correct in detail.
Assume the given answer is correct. Do not challenge or change the answer.
Please finish in one sentence.
"""
    return prompt

def wrap_llama_chat_prompt(tokenizer, user_prompt):
    messages = [
        {"role": "system", "content": "You are a helpful assistant that explains answers concisely."},
        {"role": "user", "content": user_prompt}
    ]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

def generate_openai(client, prompt, model="gpt-4", max_tokens=128):
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=max_tokens,
    )
    return resp.choices[0].message.content.strip()


def generate_gemini(client, prompt, model="gemini-2.5-flash-lite"):
    resp = client.models.generate_content(
        model=model,
        contents=prompt,
    )
    return (resp.text or "").strip()

def generate_flan_t5(model, tokenizer, prompt, max_tokens=128):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def generate_llama(model, tokenizer, prompt, max_tokens=128):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    gen_ids = outputs[0][inputs["input_ids"].shape[-1]:]
    return tokenizer.decode(gen_ids, skip_special_tokens=True).strip()

def load_ecqa_csv(path):
    df = pd.read_csv(path)
    samples = []
    for _, r in df.iterrows():
        samples.append({
            "question_text": r["q_text"],
            "answer_text": r["q_ans"]
        })
    return samples

def load_esnli_csv(path):
    df = pd.read_csv(path)
    samples = []
    for index, r in df.iterrows():
        if index == 2000:
            break
        samples.append({
            "premise": r["Sentence1"],
            "hypothesis": r["Sentence2"],
            "answer_text": r["gold_label"]
        })
    return samples

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu", type=str, help="GPU number or 'cpu'.")
    parser.add_argument("--task", choices=["ECQA", "ESNLI"], required=True)
    parser.add_argument("--seed", type=int, default=42, help="Random seed for initialization.")
    parser.add_argument("--model_name_or_path", type=str, choices=["gpt-4", "gemini-2.5-pro", "llama3.1", "flan_t5"], required=True)
    parser.add_argument("--gemini_key", type=str, required=True, help="Gemini API key")
    parser.add_argument("--gpt_key", type=str, required=True, help="GPT API key")
    parser.add_argument("--llama_token", type=str, required=True, help="Llama token from Huggingface")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--max_samples", type=int, default=None, help="Load only the first N samples for debugging")
    args = parser.parse_args()

    set_seed(args)

    # Setup device
    device = torch.device(
        f"cuda:{args.device}"
        if torch.cuda.is_available() and args.device != "cpu"
        else "cpu"
    )

    if args.model_name_or_path == "gemini-2.5-pro":
        from google import genai
        gemini_client = genai.Client(api_key=args.gemini_key)
    elif args.model_name_or_path == "gpt-4":
        from openai import OpenAI
        openai_client = OpenAI(api_key=args.gpt_key)

    current_path = os.path.dirname(os.path.abspath(__file__))

    if args.task == "ECQA":
        args.data_path = os.path.join(current_path, '../', 'dataset', args.task.lower(), 'output')
        args.data_path = os.path.normpath(args.data_path)
        args.train_file = os.path.join(args.data_path, 'ecqa_test.csv')
        samples = load_ecqa_csv(args.train_file)
    elif args.task == "ESNLI":
        args.data_path = os.path.join(current_path, '../', 'dataset', args.task.lower())
        args.data_path = os.path.normpath(args.data_path)
        args.train_file = os.path.join(args.data_path, 'esnli_test.csv')
        samples = load_esnli_csv(args.train_file)

    if args.max_samples is not None:
        samples = samples[:args.max_samples]

    hf_models = {}
    if "llama3.1" in args.model_name_or_path:
        model_id = "meta-llama/Llama-3.1-8B-Instruct"
        tokenizer = AutoTokenizer.from_pretrained(model_id, token=args.llama_token)
        model = AutoModelForCausalLM.from_pretrained(model_id, token=args.llama_token).to(device)
        hf_models["llama3.1"] = (model, tokenizer)
    if "flan_t5" in args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-large")
        model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-large").to(device)
        hf_models["flan_t5"] = (model, tokenizer)
    with open(args.output_dir, "w", encoding="utf-8") as fout:
        for ex in tqdm(samples):

            prompt = build_task_prompt(args.task, ex)

            if args.model_name_or_path == "gpt-4":
                rationale = generate_openai(openai_client, prompt, "gpt-4")

            elif args.model_name_or_path == "gemini-2.5-pro":
                rationale = generate_gemini(gemini_client, prompt)

            elif args.model_name_or_path == "llama3.1":
                model, tokenizer = hf_models["llama3.1"]
                chat_prompt = wrap_llama_chat_prompt(tokenizer, prompt)
                rationale = generate_llama(model, tokenizer, chat_prompt)

            elif args.model_name_or_path == "flan_t5":
                model, tokenizer = hf_models["flan_t5"]
                rationale = generate_flan_t5(model, tokenizer, prompt)

            if args.task == "ECQA":
                question = ex["question_text"]
            elif args.task == "ESNLI":
                question = ex["premise"] + " " + ex["hypothesis"]

            record = {
                "question_text": question,
                "answer_text": ex["answer_text"],
                "generated_rationale": rationale,
            }
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    main()
