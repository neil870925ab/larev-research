# Adapted from:
# https://github.com/HanjieChen/REV/blob/main/src/rev/rev_eval.py
import json
import tqdm
import torch
import logging
import argparse
import os
from generative import set_seed

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

from utils import init_model, load_data_ecqa, load_data_esnli, load_data_ranking


load_data_func = {
    'ECQA': load_data_ecqa,
    'ESNLI': load_data_esnli,
    'ECQA_RANKING': load_data_ranking,
    'ESNLI_RANKING': load_data_ranking,
}
special_toks = {
    'ECQA': ["[question]", "[choice]", "[answer]", "[rationale]", "<eos>", "<mask>"],
    'ESNLI': ["[premise]", "[hypothesis]", "[answer]", "[rationale]", "<eos>", "<mask>"],
}


def _fmt_penalty(p):
    # If the float is an integer (5.0) -> use '5' to match the shell script of train.sh/train_irm.sh behavior.
    # Otherwise, replace '.' with 'p' (0.2 -> '0p2')
    if p.is_integer():
        s = str(int(p))
    else:
        s = str(p).replace('.', 'p')
    return s

def normalize_pred_for_bart(pred):
    output = pred.replace("[answer]<s>", "[answer]").replace("<s>", "").strip()
    return output

def main() -> None:
    """
    Generate intensifiers and attenuators
    """
    parser = argparse.ArgumentParser()

    # Required
    parser.add_argument(
        "--model_name",
        default="t5-large",
        choices=["t5-large", "bart-large"],
        type=str,
        help="What backbone model you want to evaluate (t5-large or bart-large).",
    )

    # Optional
    parser.add_argument(
        "--max_length", default=20, type=int, required=False, help="Maximum text length"
    )
    parser.add_argument(
        "--min_length", default=1, type=int, required=False, help="Minimum text length"
    )
    parser.add_argument(
        "--k", default=0, type=int, required=False, help="k for top k sampling"
    )
    parser.add_argument(
        "--p", default=0, type=float, required=False, help="p for nucleus sampling"
    )
    parser.add_argument(
        "--beams", default=0, type=int, required=False, help="beams for beam search"
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for initialization."
    )
    parser.add_argument(
        "--temperature",
        default=1.0,
        type=float,
        required=False,
        help="temperature for sampling",
    )
    parser.add_argument(
        "--device", default="cpu", type=str, help="GPU number or 'cpu'."
    )
    parser.add_argument(
        "--task",
        default="ECQA",
        type=str,
        required=True,
        choices=["ECQA", "ESNLI"],
        help="what is the task? ECQA or ESNLI.",
    )
    parser.add_argument(
        "--task_model",
        default="t5-large",
        type=str,
        help="task model name",
    )
    parser.add_argument(
        "--split",
        default="test",
        type=str,
        help="train, val, test",
    )
    parser.add_argument(
        "--data_type",
        default="regular",
        type=str,
        help="temp: b, regular: [r, b]",
    )
    parser.add_argument(
        "--use_irm",
        default=0,
        type=int,
        help="Use IRM training or not",
    )
    parser.add_argument(
        "--irm_penalty_weight",
        default=0.2,
        type=float,
        help="IRM penalty weight",
    )
    parser.add_argument(
        "--ranking_file",
        default="",
        type=str,
        help="Input file for ranking evaluation",
    )
    parser.add_argument(
        "--ranking_result_file",
        default="ranking_result.json",
        type=str,
        help="Where to append ranking summary results (JSON list).",
    )
    parser.add_argument(
        "--ranking_type",
        default="gold",
        choices=["gold", "leaky", "gold_leaky", "vacuous", "truncated_gold_80", "truncated_gold_50", "gold_noise", "shuffled_gold"],
        type=str,
        help="Choose the type of data for ranking evaluation",
    )
    parser.add_argument(
        "--use_leak_probe",
        default=0,
        type=int,
        help="Use pretrained leakage probe ψ during training (0 / 1).",
    )
    parser.add_argument(
        "--leak_penalty_weight",
        default=0.3,
        type=float,
        help="λ_leak: weight for leakage penalty from ψ.",
    )
    parser.add_argument(
        "--task_model_type",
        default="None",
        type=str,
        choices=["None", "flan_t5", "gpt-4", "gemini-2.5-pro", "llama3.1"],
        help="Type of task model used for evaluation (None / flan_t5 / gpt-4 / gemini-2.5-pro / llama3.1).",
    )
    args = parser.parse_args()
    logger.debug(args)

    if (
        (args.k == args.p == args.beams == 0)
        or (args.k != 0 and args.p != 0)
        or (args.beams != 0 and args.p != 0)
        or (args.beams != 0 and args.k != 0)
    ):
        raise ValueError(
            "Exactly one of p, k, and beams should be set to a non-zero value."
        )

    device = torch.device(
        f"cuda:{args.device}"
        if torch.cuda.is_available() and args.device != "cpu"
        else "cpu"
    )
    logger.debug(f"Initializing {args.device}")

    # Set seed
    set_seed(args)

    
    current_path = os.path.dirname(os.path.abspath(__file__))
    if args.split == 'train':
            split_type = 'train'
    elif args.split == 'val':
            split_type = 'val'
    elif args.split == 'test':
            split_type = 'test'

    if args.split == 'ranking':
        args.data_path = os.path.join(current_path, '../', 'generate_data_for_ranking_metric', 'output', args.task)
        args.data_path = os.path.normpath(args.data_path)
        if args.task_model_type != "None":
            input_jsonl = os.path.join(args.data_path, args.task_model_type+'_'+args.task.lower()+'_'+args.ranking_file)
        else:
            input_jsonl = os.path.join(args.data_path, args.task.lower()+'_'+args.ranking_file)
        # read gold labels
        gold_temp_file = input_jsonl
        prepare_larev_eval_format_for_ranking(args, input_jsonl, args.data_path)
    else:
        # read gold labels
        gold_temp_file = os.path.join(current_path, '../', 'generate_baseline_rationales', 'output', args.task, 'baseline_rationales_'+split_type+'_output.jsonl')
        gold_temp_file = os.path.normpath(gold_temp_file)


    gold_labels = []
    with open(gold_temp_file, 'r') as json_file:
        json_list = list(json_file)
        for json_str in json_list:
                result = json.loads(json_str)
                label = result['answer_text']
                gold_labels.append(label)

    # compute the vi of rationales
    rat_outputs = compute_vi(args, current_path, data_type='regular', device=device)
    regular_model_dir = args.model_name_or_path  # save the regular model path for later use
    
    # compute the vi of baselines
    base_outputs = compute_vi(args, current_path, data_type='temp', device=device)


    rat_vi_list, base_vi_list = [], []
    base_corr_list, base_icorr_list = [], []
    rat_corr_list, rat_icorr_list = [], []
    sample_scores = []  # Store individual sample scores
    for rat_output, base_output, gold_label in zip(rat_outputs, base_outputs, gold_labels):
            rat = rat_output[0].lower().replace("[rationale]", "").replace("[answer]", "").strip()
            rat_ = base_output[0].lower().replace("[rationale]", "").replace("[answer]", "").strip()
            pred = rat_output[2][0].lower().replace("<eos>", "").replace("[answer]", "").strip() if rat_output[2] else ""
            pred_ = base_output[2][0].lower().replace("<eos>", "").replace("[answer]", "").strip() if base_output[2] else ""
            gold_label = gold_label.lower()
            
            # Calculate individual score
            individual_score = rat_output[3] - base_output[3]
            sample_scores.append({
                "input": rat_output[0],
                "phi_la_prediction": rat_output[2][0],
                "gold_label": gold_label,
                "phi_la_vi": rat_output[3],
                "phi_base_vi": base_output[3],
                "larev_score": individual_score
            })

            # compute vi no matter correct or not
            rat_vi = rat_output[3]
            rat_vi_list.append(rat_vi)
            base_vi = base_output[3]
            base_vi_list.append(base_vi)

            # compute vi for correct and incorrect predictions
            if pred == gold_label:
                    # if rat == '', set cvi = 0
                    if rat == rat_:
                         rat_corr_list.append(base_output[3])
                    else:
                         rat_corr_list.append(rat_output[3])
                    base_corr_list.append(base_output[3])
            else:
                    # show_incorrect_prediction(rat_output, base_output, gold_label) # for debug
                    if rat == rat_:
                         rat_icorr_list.append(base_output[3])
                    else:
                         rat_icorr_list.append(rat_output[3])
                    base_icorr_list.append(base_output[3])
    rat_vi_c = sum(rat_corr_list) / len(rat_corr_list)
    rat_corr_num = len(rat_corr_list)
    try:
        rat_vi_ic = sum(rat_icorr_list) / len(rat_icorr_list)
    except:
        rat_vi_ic = 0
    rat_icorr_num = len(rat_icorr_list)
    base_vi_c = sum(base_corr_list) / len(base_corr_list)
    base_corr_num = len(base_corr_list)
    try:
        base_vi_ic = sum(base_icorr_list) / len(base_icorr_list)
    except:
        base_vi_ic = 0
    base_icorr_num = len(base_icorr_list)

    # compute larev no matter correct or not
    rat_vi = sum(rat_vi_list) / len(rat_vi_list)
    base_vi = sum(base_vi_list) / len(base_vi_list)
    larev = rat_vi - base_vi

    # compute larev for correct (c) and incorrect (ic) predictions, and overall larev
    larev_c = rat_vi_c - base_vi_c
    larev_ic = rat_vi_ic - base_vi_ic

    print('Total number of examples: {}'.format(len(rat_vi_list)))
    print('Number of correct predictions: {}'.format(rat_corr_num))
    print('Number of incorrect predictions: {}'.format(rat_icorr_num))
    print('ranking type: {}'.format(args.ranking_type))
    print('larev_c: {} | larev_ic: {} | larev:{}'.format(larev_c, larev_ic, larev))

    #write the ranking summary to a file
    summary = {
        "ranking_type": args.ranking_type if args.split == 'ranking' else 'N/A',
        "larev": larev,
        "larev_c": larev_c,
        "larev_ic": larev_ic,
        "rat_corr_num": rat_corr_num,
    }

    if args.task_model_type != "None":
        ranking_result_path = os.path.join(regular_model_dir, args.task_model_type+'_'+args.ranking_result_file)
    else:
        ranking_result_path = os.path.join(regular_model_dir, args.ranking_result_file)
    os.makedirs(regular_model_dir, exist_ok=True)

    with open(ranking_result_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(summary, ensure_ascii=False) + "\n")
    
    
    # Write individual sample scores to a separate file
    if args.split == 'ranking':
        if args.task_model_type != "None":
            sample_scores_file = os.path.join(regular_model_dir, f"larev_scores_{args.task_model_type}_{args.ranking_type}.jsonl")
        else:
            sample_scores_file = os.path.join(regular_model_dir, f"larev_scores_{args.ranking_type}.jsonl")
    else:
        sample_scores_file = os.path.join(regular_model_dir, f"larev_scores_{args.split}.jsonl")
    
    with open(sample_scores_file, "w", encoding="utf-8") as f:
        for sample in sample_scores:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")




def compute_vi(args, current_path, data_type, device):
    # read gold data
    if args.task == 'ECQA':
        if data_type == 'regular':
            args.data_path = os.path.join(current_path, '../', 'dataset', 'ecqa', 'output')
            args.data_path = os.path.normpath(args.data_path)
            if args.split == 'train':
                    args.in_file = os.path.join(args.data_path, 'ecqa_train.csv')
            elif args.split == 'val':
                    args.in_file = os.path.join(args.data_path, 'ecqa_val.csv')
            elif args.split == 'test':
                    args.in_file = os.path.join(args.data_path, 'ecqa_test.csv')
            elif args.split == 'ranking':
                args.data_path = os.path.join(current_path, '../', 'generate_data_for_ranking_metric', 'output', args.task)
                args.data_path = os.path.normpath(args.data_path)
                if args.task_model_type != "None":
                    args.in_file = os.path.join(args.data_path, args.task_model_type+'_ecqa_test_regular.jsonl')
                else:
                    args.in_file = os.path.join(args.data_path, 'ecqa_test_regular.jsonl')
        elif data_type == 'temp':
            if args.split == 'ranking':
                args.data_path = os.path.join(current_path, '../', 'generate_data_for_ranking_metric', 'output', args.task)
                args.data_path = os.path.normpath(args.data_path)
                if args.task_model_type != "None":
                    args.in_file = os.path.join(args.data_path, args.task_model_type+'_ecqa_test_temp.jsonl')
                else:
                    args.in_file = os.path.join(args.data_path, 'ecqa_test_temp.jsonl')
            else:
                args.in_file  = os.path.join(current_path, '../', 'generate_baseline_rationales', 'output', args.task, 'baseline_rationales_'+args.split+'_output.jsonl')
                args.in_file = os.path.normpath(args.in_file)
    elif args.task == 'ESNLI':
        if data_type == 'regular':
            args.data_path = os.path.join(current_path, '../', 'dataset', 'esnli', 'output')
            args.data_path = os.path.normpath(args.data_path)
            if args.split == 'train':
                    args.in_file = os.path.join(args.data_path, 'esnli_train_templated.jsonl')
            elif args.split == 'val':
                    args.in_file = os.path.join(args.data_path, 'esnli_val_templated.jsonl')
            elif args.split == 'test':
                    args.in_file = os.path.join(args.data_path, 'esnli_test_templated.jsonl')
            elif args.split == 'ranking':
                args.data_path = os.path.join(current_path, '../', 'generate_data_for_ranking_metric', 'output', args.task)
                args.data_path = os.path.normpath(args.data_path)
                if args.task_model_type != "None":
                    args.in_file = os.path.join(args.data_path, args.task_model_type+'esnli_test_regular.jsonl')
                else:
                    args.in_file = os.path.join(args.data_path, 'esnli_test_regular.jsonl')
        elif data_type == 'temp':
            if args.split == 'ranking':
                args.data_path = os.path.join(current_path, '../', 'generate_data_for_ranking_metric', 'output', args.task)
                args.data_path = os.path.normpath(args.data_path)
                if args.task_model_type != "None":
                    args.in_file = os.path.join(args.data_path, args.task_model_type+'esnli_test_temp.jsonl')
                else:
                    args.in_file = os.path.join(args.data_path, 'esnli_test_temp.jsonl')
            else:
                args.in_file  = os.path.join(current_path, '../', 'generate_baseline_rationales', 'output', args.task, 'baseline_rationales_'+args.split+'_output.jsonl')
                args.in_file = os.path.normpath(args.in_file)
                
    
    if data_type == 'regular':
        if args.use_irm == 1:
            avoid_dot_in_filename_irm = _fmt_penalty(args.irm_penalty_weight) # load the transformed filename of specific model
        if args.use_leak_probe == 1:
            avoid_dot_in_filename_lp = _fmt_penalty(args.leak_penalty_weight) # load the transformed filename of specific model

        if args.use_irm == 1 and args.use_leak_probe == 1:
            args.model_name_or_path = os.path.join(
                current_path,
                'output',
                f"{args.task}_{data_type}_irm-{args.model_name}-penalty_{avoid_dot_in_filename_irm}-leak_probe_penalty_{avoid_dot_in_filename_lp}"
            )
        elif args.use_irm == 1 and args.use_leak_probe == 0:
            args.model_name_or_path = os.path.join(
                current_path,
                'output',
                f"{args.task}_{data_type}_irm-{args.model_name}-penalty_{avoid_dot_in_filename_irm}"
            )
        elif args.use_irm == 0 and args.use_leak_probe == 1:
            args.model_name_or_path = os.path.join(
                current_path,
                'output',
                f"{args.task}_{data_type}-{args.model_name}-leak_probe_penalty_{avoid_dot_in_filename_lp}"
            )
        elif args.use_irm == 0 and args.use_leak_probe == 0:
            args.model_name_or_path = os.path.join(current_path, 'output', args.task+'_'+data_type+'-'+args.model_name) #standard REV Regular Model from REV paper
        
    
    if data_type == 'temp':
        args.model_name_or_path = os.path.join(current_path, 'output', args.task+'_'+data_type+'-'+args.model_name) #standard REV Temp Model from REV paper
    
    if not os.path.exists(args.model_name_or_path):
        raise ValueError(f"Error: Model path '{args.model_name_or_path}' does not exist!")
    logger.info(f"Loading model from {args.model_name_or_path}")

    tokenizer, model = init_model(args.model_name_or_path, device)

    args.task = transforming_task_type(args)
    examples = load_data_func[args.task](args, args.in_file, data_type=data_type, shuffle=False)
    args.task = args.task.replace('_RANKING', '')  # reset back to the original task name
    
    # examples = examples[:1]   # for debug
    
    output_list = []
    if data_type == 'regular':
        for input, output in tqdm.tqdm(examples):
            try:
                info, preds = generate_conditional(
                    tokenizer,
                    model,
                    args,
                    input,
                    output,
                    device,
                )

            except Exception as exp:
                logger.info(exp)
                preds = []

            output_list.append((input, output, preds, info.item()))
    elif data_type == 'temp':
        for input, output in tqdm.tqdm(examples):
            try:
                info, preds = generate_conditional(
                    tokenizer,
                    model,
                    args,
                    input,
                    output,
                    device,
                )

            except Exception as exp:
                logger.info(exp)
                preds = []

            output_list.append((input, output, preds, info.item()))
    else:
        raise ValueError("data_type should be either 'regular' or 'temp'!")
    return output_list

def show_incorrect_prediction(rat_output, base_output, gold_label): # only watch the incorrect predictions from regular evaluator, not temp evaluator
    print('-' * 20)
    print('input: {}'.format(rat_output[0]))
    print('gold label: {}'.format(gold_label))
    print('pred label: {}'.format(rat_output[2][0].lower().replace("<eos>", "").replace("[answer]", "").strip()))
    print('v-info: {}'.format(rat_output[3]))
    print('-' * 20)



def prepare_larev_eval_format_for_ranking(args, input_jsonl, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    task = args.task.lower()
    if args.task_model_type != "None":
        regular_path = os.path.join(out_dir, f"{args.task_model_type}_{task}_test_regular.jsonl")
        temp_path = os.path.join(out_dir, f"{args.task_model_type}_{task}_test_temp.jsonl")
    else:
        regular_path = os.path.join(out_dir, f"{task}_test_regular.jsonl")
        temp_path = os.path.join(out_dir, f"{task}_test_temp.jsonl")

    with open(input_jsonl, "r", encoding="utf-8") as f_in, \
         open(regular_path, "w", encoding="utf-8") as f_reg, \
         open(temp_path, "w", encoding="utf-8") as f_temp:
        
        data = [json.loads(line) for line in f_in]

        for i, ex in enumerate(data):
            q, a = ex["question_text"], ex["answer_text"]
            gold = ex.get("gold_rationale", "")
            gold_leaky = ex.get("gold_leaky_rationale", "")
            vacuous = ex.get("vacuous_rationale")
            leaky = ex.get("leaky_rationale", "")
            truncated_gold_80 = ex.get("truncated_gold_80_rationale", "")
            truncated_gold_50 = ex.get("truncated_gold_50_rationale", "")
            gold_plus_noise = ex.get("gold_noise_rationale", "")
            shuffled_rationale = ex.get("shuffled_gold_rationale", "")
            

            # --- regular (R,B) pairs ---
            pairs = [
                ("gold", gold, vacuous),
                ("leaky", leaky, vacuous),
                ("gold_leaky", gold_leaky, vacuous),
                ("vacuous", vacuous, vacuous),
                ("truncated_gold_80", truncated_gold_80, vacuous),
                ("truncated_gold_50", truncated_gold_50, vacuous),
                ("gold_noise", gold_plus_noise, vacuous),
                ("shuffled_gold", shuffled_rationale, vacuous),
            ]

            for tag, R, B in pairs:
                reg_item = {
                    "type": tag,
                    "question": q,
                    "answer_text": a,
                    "rationale": R,
                    "baseline_rationale": B
                }
                tmp_item = {
                    "type": tag,
                    "question": q,
                    "answer_text": a,
                    "baseline_rationale": B
                }
                f_reg.write(json.dumps(reg_item, ensure_ascii=False) + "\n")
                f_temp.write(json.dumps(tmp_item, ensure_ascii=False) + "\n")

    if not (os.path.exists(regular_path) and os.path.exists(temp_path)):
        raise ValueError("Error in preparing larev eval format for ranking!")
    
def transforming_task_type(args):
    if args.split == 'ranking':
        return args.task + '_RANKING'
    else:
        return args.task

def generate_conditional(tokenizer, model, args, input, output, device):
    """
    Generate a sequence with models like Bart and T5
    """
    tokens = tokenizer.tokenize(input)
    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    decoder_start_token_id = input_ids[-1]

    input_ids = torch.tensor([input_ids]).to(device)
    max_length = args.max_length
    min_length = args.min_length

    outputs = model.generate(
        input_ids,
        do_sample=args.beams == 0,
        max_length=max_length,
        min_length=min_length,
        temperature=args.temperature,
        top_p=args.p if args.p > 0 else None,
        top_k=args.k if args.k > 0 else None,
        num_beams=args.beams if args.beams is not None else 1,
        decoder_start_token_id = decoder_start_token_id,
        no_repeat_ngram_size=2,
        return_dict_in_generate=True,
        eos_token_id=tokenizer.eos_token_id,
        num_return_sequences=1
    )

    preds = [tokenizer.decode(
        out, skip_special_tokens=False, clean_up_tokenization_spaces=False) for out in outputs.sequences]
    
    if 'bart' in args.model_name_or_path.lower():
        preds = [normalize_pred_for_bart(pred) for pred in preds]

    output_ids = [input_ids[0][-1].item()] + tokenizer.convert_tokens_to_ids(tokenizer.tokenize(output))
    output_ids = torch.tensor([output_ids]).to(device)
    decoder_input_ids = output_ids[:, :-1].contiguous()

    with torch.no_grad():
        lm_logits = model(input_ids, decoder_input_ids=decoder_input_ids, use_cache=False)[0]
    num_choices, out_length, vocab_size = lm_logits.shape
    lm_labels = output_ids[:, 1:].clone().contiguous()
    m = torch.nn.Softmax(dim=2)
    probs = m(lm_logits).view(-1, vocab_size)
    log_probs = [torch.log(probs[n][l]) for n, l in enumerate(lm_labels.view(-1))]

    return sum(log_probs) / len(log_probs), preds


if __name__ == "__main__":
    main()
