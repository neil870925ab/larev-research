import pandas as pd
import json
import os
import argparse
import random
import numpy as np

def truncate_rationale(text, ratio):
    """Keep first ratio portion of tokens."""
    tokens = text.split()
    keep = max(1, int(len(tokens) * ratio))
    return " ".join(tokens[:keep])

def shuffle_tokens(text):
    """Generate random-token rationale by shuffling the original tokens."""
    tokens = text.split()
    random.shuffle(tokens)
    return " ".join(tokens)

def insert_noise_interleaved(text, vocab, noise_ratio=0.2):
    """
    Insert random noise tokens into the rationale.
    noise_ratio: fraction of positions to inject noise.
    """
    tokens = text.split()
    new_tokens = []

    for t in tokens:
        new_tokens.append(t)

        if random.random() < noise_ratio:
            noise_token = random.choice(vocab)
            new_tokens.append(noise_token)

    return " ".join(new_tokens)

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, required=True, choices=["ECQA", "ESNLI"], 
                        help="Task name (ECQA or ESNLI)")
    parser.add_argument("--out_dir", type=str, required=True, help="Path to output JSONL file")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for initialization.")
    parser.add_argument("--evaluate_on_task_model", action="store_true", help="Whether to use task model generated rationales to build vocabulary")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    if not args.evaluate_on_task_model:

        current_path = os.path.dirname(os.path.abspath(__file__))
        if args.task == "ECQA":
            data_path = os.path.join(current_path, '../', 'dataset', args.task.lower(), 'output')
            data_path = os.path.normpath(data_path)
            test_file = os.path.join(data_path, f'{args.task.lower()}_test.csv')
        elif args.task == "ESNLI":
            data_path = os.path.join(current_path, '../', 'dataset', args.task.lower())
            data_path = os.path.normpath(data_path)
            test_file = os.path.join(data_path, f'{args.task.lower()}_test.csv')
        data_path = os.path.join(current_path, '../', 'generate_baseline_rationales', 'output', args.task, 'baseline_rationales_test_output.jsonl')
        data_path = os.path.normpath(data_path)
        baseline_rationales_file = pd.read_json(data_path, lines=True)
        if not os.path.exists(test_file):
            raise FileNotFoundError(f"Test file csv not found: {test_file}")
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Baseline rationales file not found: {data_path}")
        
        df = pd.read_csv(test_file)
        
        # build vocabulary from baseline rationales and gold rationales
        vocab = set()
        # 1. Update vocabulary from baseline JSONL (vacuous rationale)
        for text in baseline_rationales_file['question_statement_text']:
            if isinstance(text, str):
                vocab.update(text.split())

        # 2. Update vocabulary from gold rationales
        if args.task == "ECQA":
            for text in df['taskB']:
                if isinstance(text, str):
                    vocab.update(text.replace("\n", " ").split())
        elif args.task == "ESNLI":
            for text in df['Explanation_1']:
                if isinstance(text, str):
                    vocab.update(text.replace("\n", " ").split())
        vocab = list(vocab)

        results = []
        if args.task == "ECQA":
            for i, row in df.iterrows():
                if i >= len(baseline_rationales_file):
                    raise ValueError(f"Index {i} exceeds baseline rationales file length {len(baseline_rationales_file)}")
                question = row['q_text'].replace('\n', " ")
                gold = row['taskB'].replace('\n', " ")
                answer = row['q_ans']
                leaky = f"The answer is {answer}."
                gold_leaky = f"{gold} {leaky}"
                vacuous = baseline_rationales_file.iloc[i]['question_statement_text']

                shuffled_rationale = shuffle_tokens(gold)
                truncated_gold_80 = truncate_rationale(gold, 0.8)
                truncated_gold_50 = truncate_rationale(gold, 0.5)
                gold_plus_noise = insert_noise_interleaved(gold, vocab, noise_ratio=0.2)

                results.append({
                    "question_text": question,
                    "answer_text": answer,
                    "gold_rationale": gold,
                    "leaky_rationale": leaky,
                    "gold_leaky_rationale": gold_leaky,
                    "vacuous_rationale": vacuous,
                    "truncated_gold_80_rationale": truncated_gold_80,
                    "truncated_gold_50_rationale": truncated_gold_50,
                    "gold_noise_rationale": gold_plus_noise,
                    "shuffled_gold_rationale": shuffled_rationale,
                })
        elif args.task == "ESNLI":
            for i, row in df.iterrows():
                if i == 2000: #we only use first 2000 samples for esnli test set
                    break
                question = baseline_rationales_file.iloc[i]['question_text']
                gold = row['Explanation_1'].replace('\n', " ")
                answer = row['gold_label']
                leaky = f"The answer is {answer}."
                gold_leaky = f"{gold} {leaky}"
                vacuous = baseline_rationales_file.iloc[i]['question_statement_text']

                shuffled_rationale = shuffle_tokens(gold)
                truncated_gold_80 = truncate_rationale(gold, 0.8)
                truncated_gold_50 = truncate_rationale(gold, 0.5)
                gold_plus_noise = insert_noise_interleaved(gold, vocab, noise_ratio=0.2)

                results.append({
                    "question_text": question,
                    "answer_text": answer,
                    "gold_rationale": gold,
                    "leaky_rationale": leaky,
                    "gold_leaky_rationale": gold_leaky,
                    "vacuous_rationale": vacuous,
                    "truncated_gold_80_rationale": truncated_gold_80,
                    "truncated_gold_50_rationale": truncated_gold_50,
                    "gold_noise_rationale": gold_plus_noise,
                    "shuffled_gold_rationale": shuffled_rationale,
                })

        if not os.path.exists(args.out_dir):
            os.makedirs(args.out_dir, exist_ok=True)
            print("Directory '% s' created" % args.out_dir)
        out_file = os.path.join(args.out_dir, f"{args.task.lower()}_test_data_for_ranking_metric.jsonl")
        with open(out_file, 'w', encoding='utf-8') as f:
            for r in results:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

        print(f"Saved {len(results)} samples to {out_file}")

    else:

        task_model = ["flan_t5", "gpt-4", "gemini-2.5-pro", "llama3.1"]

        for model in task_model:

            current_path = os.path.dirname(os.path.abspath(__file__))
            if args.task == "ECQA":
                data_path = os.path.join(current_path, '../', 'generate_rationale_from_task_model', 'output', args.task, model)
                data_path = os.path.normpath(data_path)
                test_file = os.path.join(data_path, 'generated_rationale.jsonl')
            elif args.task == "ESNLI":
                data_path = os.path.join(current_path, '../', 'generate_rationale_from_task_model', 'output', args.task, model)
                data_path = os.path.normpath(data_path)
                test_file = os.path.join(data_path, 'generated_rationale.jsonl')
            data_path = os.path.join(current_path, '../', 'generate_baseline_rationales', 'output', args.task, 'baseline_rationales_test_output.jsonl')
            data_path = os.path.normpath(data_path)
            baseline_rationales_file = pd.read_json(data_path, lines=True)
            if not os.path.exists(test_file):
                raise FileNotFoundError(f"Test file csv not found: {test_file}")
            if not os.path.exists(data_path):
                raise FileNotFoundError(f"Baseline rationales file not found: {data_path}")
            
            df = pd.read_json(test_file, lines=True)
            
            # build vocabulary from baseline rationales and gold rationales
            vocab = set()
            # 1. Update vocabulary from baseline JSONL (vacuous rationale)
            for text in baseline_rationales_file['question_statement_text']:
                if isinstance(text, str):
                    vocab.update(text.split())

            # 2. Update vocabulary from gold rationales
            if args.task == "ECQA":
                for text in df['generated_rationale']:
                    if isinstance(text, str):
                        vocab.update(text.replace("\n", " ").split())
            elif args.task == "ESNLI":
                for text in df['generated_rationale']:
                    if isinstance(text, str):
                        vocab.update(text.replace("\n", " ").split())
            vocab = list(vocab)

            results = []
            if args.task == "ECQA":
                for i, row in df.iterrows():
                    if i >= len(baseline_rationales_file):
                        raise ValueError(f"Index {i} exceeds baseline rationales file length {len(baseline_rationales_file)}")
                    question = row['question_text'].replace('\n', " ")
                    gold = row['generated_rationale'].replace('\n', " ")
                    answer = row['answer_text']
                    leaky = f"The answer is {answer}."
                    gold_leaky = f"{gold} {leaky}"
                    vacuous = baseline_rationales_file.iloc[i]['question_statement_text']

                    shuffled_rationale = shuffle_tokens(gold)
                    truncated_gold_80 = truncate_rationale(gold, 0.8)
                    truncated_gold_50 = truncate_rationale(gold, 0.5)
                    gold_plus_noise = insert_noise_interleaved(gold, vocab, noise_ratio=0.2)

                    results.append({
                        "question_text": question,
                        "answer_text": answer,
                        "gold_rationale": gold,
                        "leaky_rationale": leaky,
                        "gold_leaky_rationale": gold_leaky,
                        "vacuous_rationale": vacuous,
                        "truncated_gold_80_rationale": truncated_gold_80,
                        "truncated_gold_50_rationale": truncated_gold_50,
                        "gold_noise_rationale": gold_plus_noise,
                        "shuffled_gold_rationale": shuffled_rationale,
                    })
            elif args.task == "ESNLI":
                for i, row in df.iterrows():
                    if i == 2000: #we only use first 2000 samples for esnli test set
                        break
                    question = baseline_rationales_file.iloc[i]['question_text']
                    gold = row['generated_rationale'].replace('\n', " ")
                    answer = row['answer_text']
                    leaky = f"The answer is {answer}."
                    gold_leaky = f"{gold} {leaky}"
                    vacuous = baseline_rationales_file.iloc[i]['question_statement_text']

                    shuffled_rationale = shuffle_tokens(gold)
                    truncated_gold_80 = truncate_rationale(gold, 0.8)
                    truncated_gold_50 = truncate_rationale(gold, 0.5)
                    gold_plus_noise = insert_noise_interleaved(gold, vocab, noise_ratio=0.2)

                    results.append({
                        "question_text": question,
                        "answer_text": answer,
                        "gold_rationale": gold,
                        "leaky_rationale": leaky,
                        "gold_leaky_rationale": gold_leaky,
                        "vacuous_rationale": vacuous,
                        "truncated_gold_80_rationale": truncated_gold_80,
                        "truncated_gold_50_rationale": truncated_gold_50,
                        "gold_noise_rationale": gold_plus_noise,
                        "shuffled_gold_rationale": shuffled_rationale,
                    })
                
            if not os.path.exists(args.out_dir):
                os.makedirs(args.out_dir, exist_ok=True)
                print("Directory '% s' created" % args.out_dir)
            out_file = os.path.join(args.out_dir, f"{model}_{args.task.lower()}_test_data_for_ranking_metric.jsonl")
            with open(out_file, 'w', encoding='utf-8') as f:
                for r in results:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")

            print(f"Saved {len(results)} samples to {out_file}")
    



if __name__ == "__main__":
    main()

