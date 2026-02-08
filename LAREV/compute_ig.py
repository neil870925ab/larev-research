import json
import argparse
import torch
import os
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from generative import set_seed

def generate_conditional(args, tokenizer, model, input, device):
    """
    Generate a sequence with models like Bart and T5
    """
    tokens = tokenizer.tokenize(input)
    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    decoder_start_token_id = tokenizer.convert_tokens_to_ids("[answer]")
    if decoder_start_token_id is None or decoder_start_token_id == tokenizer.unk_token_id:
        raise ValueError(
            "Tokenizer does not recognize '[answer]'. "
            "Make sure you have added special tokens during training AND saved tokenizer in model_dir."
        )

    input_ids = torch.tensor([input_ids]).to(device)

    outputs = model.generate(
        input_ids,
        max_length=64,
        decoder_start_token_id = decoder_start_token_id,
        return_dict_in_generate=True,
        eos_token_id=tokenizer.eos_token_id,
        num_return_sequences=1
    )

    preds = [tokenizer.decode(
        out, skip_special_tokens=False, clean_up_tokenization_spaces=False) for out in outputs.sequences]
    
    if 'bart' in args.model_name_or_path.lower():
        preds = [normalize_pred_for_bart(pred) for pred in preds]
    
    return preds

def normalize_pred_for_bart(pred):
    output = pred.replace("[answer]<s>", "[answer]").replace("<s>", "").strip()
    return output

def ig(
    model,
    tokenizer,
    source_text,
    target_text,
    steps=32,
    device="cuda",
    max_source_length=256,
    max_target_length=64,
):
    """
    Returns:
    - token_scores: numpy array, length = subword length (including special tokens)
    - enc: tokenizer encoding (with offset_mapping etc)
    """

    model.eval()
    model.to(device)

    enc = tokenizer(
        source_text,
        truncation=True,
        padding=False,
        max_length=max_source_length,
        return_offsets_mapping=True,
        return_tensors="pt",
    )
    input_ids = enc["input_ids"].to(device)          # [1, L]
    attention_mask = enc["attention_mask"].to(device)
    offsets = enc["offset_mapping"][0].tolist()      # [(start, end), ...], len = L

    tgt = tokenizer(
        target_text,
        truncation=True,
        padding=False,
        max_length=max_target_length,
        return_tensors="pt",
    )
    labels = tgt["input_ids"].to(device)             # [1, T]

    # get encoder input embeddings (this is x in IG)
    emb_layer = model.get_input_embeddings()         # shared embedding
    emb_input = emb_layer(input_ids)                # [1, L, D]

    # baseline: all-zero embedding
    baseline = torch.zeros_like(emb_input)

    diff = emb_input - baseline
    total_grad = torch.zeros_like(emb_input)

    for i in range(1, steps + 1):
        alpha = float(i) / steps
        emb_step = baseline + alpha * diff
        emb_step.requires_grad_(True)

        outputs = model(
            inputs_embeds=emb_step,
            attention_mask=attention_mask,
            labels=labels,
        )
        loss = outputs.loss  # scalar

        grads = torch.autograd.grad(loss, emb_step)[0]   # [1, L, D]
        total_grad += grads

    avg_grad = total_grad / steps

    # Apply IG formula: (x - x0) * avg_grad, sum over hidden_size dimension → token attribution
    ig = (diff * avg_grad).sum(dim=-1).squeeze(0)        # [L]
    ig = (ig * attention_mask.squeeze(0)).detach().cpu().numpy()

    return ig, enc

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, required=True, help="Path to the fine-tuned model which will be used to compute IG")
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--task", type=str, required=True, default="ECQA", choices=["ECQA", "ESNLI"], 
                        help="Task name (ECQA or ESNLI)")
    parser.add_argument("--model_name_or_path", type=str, required=True, default="t5-large", choices=["t5-large", "bart-large"])
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--text_field", type=str, default="question_statement_text")
    parser.add_argument("--label_field", type=str, default="answer_text")
    parser.add_argument("--max_source_length", type=int, default=256)
    parser.add_argument("--max_target_length", type=int, default=64)
    parser.add_argument("--mask_token", type=str, default="<mask>")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--do_prediction", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Set seed
    set_seed(args)

    if not os.path.exists(args.model_dir):
        raise FileNotFoundError(f"Model directory not found: {args.model_dir}")

    device = torch.device(
        f"cuda:{args.device}"
        if torch.cuda.is_available() and args.device != "cpu"
        else "cpu"
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, use_fast=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_dir)
    model.to(device)
    model.eval()

    SPLIT = ['train', 'val', 'test']

    for split in SPLIT:
        current_path = os.path.dirname(os.path.abspath(__file__))
        args.input_file = os.path.join(current_path, '../', 'generate_baseline_rationales', 'output', args.task, f'baseline_rationales_{split}_output.jsonl')
        args.input_file = os.path.normpath(args.input_file)
        args.output_file = os.path.join(current_path, '../', 'generate_baseline_rationales', 'output', args.task, f'baseline_rationales_{split}_output_ig_{args.model_name_or_path}.jsonl')
        args.output_file = os.path.normpath(args.output_file)
        if not os.path.exists(args.input_file):
            raise FileNotFoundError(f"Input file not found: {args.input_file}")
        os.makedirs(os.path.dirname(args.output_file), exist_ok=True)

        fout = open(args.output_file, "w", encoding="utf-8")

        with open(args.input_file, "r", encoding="utf-8") as f:
            total_lines = sum(1 for _ in f)

        if args.max_samples is not None:
            total_lines = min(total_lines, args.max_samples)

        error_indexes = []

        with open(args.input_file, "r", encoding="utf-8") as fin:
            for idx, line in enumerate(tqdm(fin, desc=f"IG attribution for {split} file", total=total_lines), start=0):

                if args.max_samples and idx >= args.max_samples:
                    break

                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)

                src_raw = obj.get(args.text_field, None)
                ans_raw = obj.get(args.label_field, None)

                if src_raw is None or ans_raw is None:
                    raise ValueError(f"Missing text or answer at sample {idx}")

                source_text = f"[rationale] {src_raw} [answer]"
                answer_text = f"[answer] {ans_raw} <eos>"

                if not source_text or not answer_text:
                    raise ValueError(f"Missing text or answer at sample {idx}")

                try:
                    ig_scores, enc = ig(
                        model=model,
                        tokenizer=tokenizer,
                        source_text=source_text,
                        target_text=answer_text,
                        steps=args.steps,
                        device=device,
                        max_source_length=args.max_source_length,
                        max_target_length=args.max_target_length,
                    )
                except RuntimeError as e:
                    print(f"RuntimeError at sample {idx}: {e}")
                    obj["most_leaky_term"] = None
                    obj["question_statement_text_masked"] = None
                    fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
                    error_indexes.append(idx)
                    continue

                offsets = enc["offset_mapping"][0].tolist()
                ig_full = ig_scores
                orig_text = obj.get(args.text_field, "")
                prefix_start = source_text.find(orig_text)
                prefix_end = prefix_start + len(orig_text)
                # tokens = tokenizer.convert_ids_to_tokens(enc["input_ids"][0])

                # skip offset = (0,0) special tokens
                valid_scores = []
                valid_offsets = []
                for (s, e), score in zip(offsets, ig_full):
                    if (s, e) == (0, 0):
                        continue
                    if s < prefix_start or e > prefix_end:
                        continue
                    span = source_text[s:e]
                    # if the span contains no alphanumeric characters, skip it. For example, commas or spaces.
                    if not any(ch.isalnum() for ch in span):
                        continue
                    valid_scores.append(score)
                    valid_offsets.append((s, e))

                if len(valid_scores) == 0:
                    raise ValueError(f"No valid tokens found at sample {idx}")

                # get the subword with the highest attribution and map it back to the original string's char span
                top_idx = int(np.argmin(valid_scores))
                char_start, char_end = valid_offsets[top_idx]
                inner_start = max(char_start - prefix_start, 0)
                inner_end = max(char_end - prefix_start, 0)
                if inner_start >= len(orig_text):
                    inner_start = len(orig_text) - 1
                if inner_end > len(orig_text):
                    inner_end = len(orig_text)

                text = orig_text
                s = inner_start
                e = inner_end

                while s > 0 and text[s - 1].isalnum():
                    s -= 1
                while e < len(text) and text[e].isalnum():
                    e += 1

                word_span = text[s:e].strip()
                if word_span == "":
                    obj["most_leaky_term"] = ""
                    obj["question_statement_text_masked"] = orig_text
                    fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
                    error_indexes.append(idx)
                    continue

                masked_text = text[:s] + args.mask_token + text[e:]

                obj["most_leaky_term"] = word_span
                obj["question_statement_text_masked"] = masked_text

                if args.do_prediction:

                    with torch.no_grad():
                        pred_text = generate_conditional(
                            args,
                            tokenizer,
                            model,
                            source_text,
                            device
                        )

                    obj["model_prediction"] = pred_text

                fout.write(json.dumps(obj, ensure_ascii=False) + "\n")

        fout.close()
        print(f"Saved IG annotation to {args.output_file}")
        if error_indexes:
            print(f"Samples with empty spans: {error_indexes}")

if __name__ == "__main__":
    main()
