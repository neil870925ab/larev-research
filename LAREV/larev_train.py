# Adapted from:
# https://github.com/HanjieChen/REV/blob/main/src/rev/rev_train.py
import os
import torch
import logging
import argparse

from torch.utils.data import Dataset
from collections import OrderedDict

from utils import init_model, load_data_ecqa, load_data_esnli, load_data_ecqa_irm, load_data_esnli_irm
from generative import evaluate, train, set_seed

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

load_data_func = {
    'ECQA': load_data_ecqa,
    'ESNLI': load_data_esnli,
    'ECQA_IRM': load_data_ecqa_irm,
    'ESNLI_IRM': load_data_esnli_irm,
}
special_toks = {
    'ECQA': ["[question]", "[choice]", "[answer]", "[rationale]", "<eos>", "<mask>"],
    'ESNLI': ["[premise]", "[hypothesis]", "[answer]", "[rationale]", "<eos>", "<mask>"],
}

def debug_print_dataloader_samples(tokenizer, dataset, n_samples=3, batch_size=3):
    """
    Print tokens, ids and decoded text for a few samples from the dataset.
    """
    from torch.utils.data import DataLoader
    dl = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    printed = 0
    for batch in dl:
        bsz = batch["inputs"].size(0)
        for i in range(bsz):
            if printed >= n_samples:
                return
            input_ids = batch["inputs"][i].tolist()
            input_mask = batch["input_mask"][i].tolist()
            output_ids = batch["outputs"][i].tolist()
            output_mask = batch["output_mask"][i].tolist()

            # trim by mask
            in_len = sum(int(x) for x in input_mask)
            out_len = sum(int(x) for x in output_mask)
            in_ids_trim = input_ids[:in_len]
            out_ids_trim = output_ids[:out_len]

            print(f"--- Sample {printed} ---")
            print("input_ids:", in_ids_trim)
            print("input_tokens:", tokenizer.convert_ids_to_tokens(in_ids_trim))
            try:
                print("input_decoded:", tokenizer.decode(in_ids_trim, skip_special_tokens=False))
            except Exception:
                print("input_decoded: <decode error>")

            print("output_ids:", out_ids_trim)
            print("output_tokens:", tokenizer.convert_ids_to_tokens(out_ids_trim))
            try:
                print("output_decoded:", tokenizer.decode(out_ids_trim, skip_special_tokens=False))
            except Exception:
                print("output_decoded: <decode error>")

            printed += 1

class EncoderDecoderTextDataset(Dataset):
    def __init__(self, tokenizer, args, file_path, block_size=512, is_train=True):
        examples = load_data_func[args.task](args, file_path, data_type=args.data_type, shuffle=False)  # Don't shuffle here
        logger.info(examples[:3])

        self.inputs = []
        self.outputs = []
        self.input_lengths = []
        self.output_lengths = []
        self.inputs_leak = []   #for leakage probe
        self.input_leak_lengths = [] #for leakage probe
        self.question_ids = [] #for irm
        self.env_ids = [] #for irm

        # For IRM datasets, group samples by question
        if args.use_irm == 1 and is_train:
            # Assume data format: each question has 3 consecutive environments (E1, E2, E3)
            for ex in examples:
                if len(ex) == 5:
                    source_text, target_text, q_id, env_id, leak_source_text = ex
                else:
                    raise ValueError("For IRM training, each example must have 5 elements: source_text, target_text, question_id, environment_id, leak_source_text")
                
                # Process the example
                input_enc = tokenizer(
                    source_text,
                    max_length=args.max_input_length,
                    padding='max_length',
                    truncation=True,
                    return_tensors='pt',
                    add_special_tokens=False,
                )
                output_enc = tokenizer(
                    target_text,
                    max_length=args.max_length,
                    padding='max_length',
                    truncation=True,
                    return_tensors='pt',
                    add_special_tokens=False,
                )

                leak_enc = tokenizer(
                    leak_source_text,
                    max_length=args.max_input_length,
                    padding='max_length',
                    truncation=True,
                    return_tensors='pt',
                    add_special_tokens=False,
                )

                self.inputs.append(input_enc['input_ids'].squeeze(0))
                self.outputs.append(output_enc['input_ids'].squeeze(0))
                self.input_lengths.append((input_enc['attention_mask'] == 1).sum().item())
                self.output_lengths.append((output_enc['attention_mask'] == 1).sum().item())
                self.question_ids.append(q_id)
                self.env_ids.append(env_id)

                if args.use_leak_probe == 1:
                    self.inputs_leak.append(leak_enc['input_ids'].squeeze(0))
                    self.input_leak_lengths.append((leak_enc['attention_mask'] == 1).sum().item())
                else:
                    self.inputs_leak = None
                    self.input_leak_lengths = None

            logger.info(f"Total questions: {len(set(self.question_ids))}, total examples: {len(self.inputs)}")
          
        else:
            # Regular processing for non-IRM datasets or IRM eval datasets
            for ex in examples:
                if len(ex) == 3:
                    source_text, target_text, leak_source_text = ex
                else:
                    raise ValueError("Each example must have 3 elements: source_text, target_text, leak_source_text")
                
                input_enc = tokenizer(
                    source_text,
                    max_length=args.max_input_length,
                    padding='max_length',
                    truncation=True,
                    return_tensors='pt',
                    add_special_tokens=False,
                )
                output_enc = tokenizer(
                    target_text,
                    max_length=args.max_length,
                    padding='max_length',
                    truncation=True,
                    return_tensors='pt',
                    add_special_tokens=False,
                )
                leak_enc = tokenizer(
                    leak_source_text,
                    max_length=args.max_input_length,
                    padding='max_length',
                    truncation=True,
                    return_tensors='pt',
                    add_special_tokens=False,
                )

                self.inputs.append(input_enc['input_ids'].squeeze(0))
                self.outputs.append(output_enc['input_ids'].squeeze(0))
                self.input_lengths.append((input_enc['attention_mask'] == 1).sum().item())
                self.output_lengths.append((output_enc['attention_mask'] == 1).sum().item())
                self.question_ids = None
                self.env_ids = None
                if args.use_leak_probe == 1:
                    self.inputs_leak.append(leak_enc['input_ids'].squeeze(0))
                    self.input_leak_lengths.append((leak_enc['attention_mask'] == 1).sum().item())
                else:
                    self.inputs_leak = None
                    self.input_leak_lengths = None

            logger.info(f"Total questions: {len(examples)}, total examples: {len(self.inputs)}")

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, item):
        data = {
            "inputs": self.inputs[item],
            "input_mask": torch.tensor([1]*self.input_lengths[item] + [0]*(len(self.inputs[item])-self.input_lengths[item])),
            "outputs": self.outputs[item],
            "output_mask": torch.tensor([1]*self.output_lengths[item] + [0]*(len(self.outputs[item])-self.output_lengths[item])),
        }
        if self.question_ids is not None and self.env_ids is not None:  # irm
            data["question_ids"] = torch.tensor(self.question_ids[item])
            data["env_ids"] = torch.tensor(self.env_ids[item])
        if self.inputs_leak is not None: # leakage probe
            data["inputs_leak"] = self.inputs_leak[item]
            data["input_leak_mask"] = torch.tensor(
                [1] * self.input_leak_lengths[item]
                + [0] * (len(self.inputs_leak[item]) - self.input_leak_lengths[item])
            )

        return data

def transforming_task_type(args):
    if args.data_type == 'filtered':
        return args.task + '_FILTERED'
    elif args.use_irm == 1:
        return args.task + '_IRM'
    else:
        return args.task


def main():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument(
        "--out_dir",
        default=None,
        type=str,
        required=True,
        help="Out directory for checkpoints.",
    )

    # Other parameters
    parser.add_argument(
        "--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer."
    )
    parser.add_argument(
        "--device", default="cpu", type=str, help="GPU number or 'cpu'."
    )
    parser.add_argument(
        "--do_eval", action="store_true", help="Whether to run eval on the dev set."
    )
    parser.add_argument(
        "--do_lower_case",
        action="store_true",
        help="Set this flag if you are using an uncased model.",
    )
    parser.add_argument(
        "--do_train", action="store_true", help="Whether to run training."
    )
    parser.add_argument(
        "--eval_batch_size", default=8, type=int, help="Batch size for evaluation."
    )
    parser.add_argument(
        "--eval_during_train",
        action="store_true",
        help="Evaluate at each train logging step.",
    )
    parser.add_argument(
        "--save_eval_predictions_during_train",
        action="store_true",
        help="Save eval predictions during training when finding a better model.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=8,
        help="Steps before backward pass.",
    )
    parser.add_argument(
        "--learning_rate",
        default=3e-5,
        type=float,
        help="The initial learning rate for Adam.",
    )
    parser.add_argument(
        "--logging_steps",
        type=int,
        default=1000,
        help="Log every X updates steps (default after each epoch).",
    )
    parser.add_argument(
        "--max_input_length",
        default=128,
        type=int,
        help="Maximum input event length in words.",
    )
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
        "--temperature",
        default=1.0,
        type=float,
        required=False,
        help="temperature for sampling",
    )
    parser.add_argument(
        "--max_grad_norm", default=1.0, type=float, help="Max gradient norm."
    )
    parser.add_argument(
        "--max_steps",
        default=-1,
        type=int,
        help="If > 0: total number of training steps to perform.",
    )
    parser.add_argument(
        "--model_name_or_path",
        default="t5-large",
        choices=["t5-large", "bart-large"],
        type=str,
        help="Backbone model name or path (t5-large or bart-large).",
    )
    parser.add_argument(
        "--num_train_epochs",
        default=8.0, #To reproduce the results in the thesie, we train for 8 epochs for non-IRM models (e.g., phi and phi base) and 2 epochs for IRM models (e.g., phi LA).
        type=float,
        help="Number of training epochs to perform.",
    )
    parser.add_argument(
        "--overwrite_cache", action="store_true", help="Overwrite the cached data."
    )
    parser.add_argument(
        "--overwrite_out_dir",
        action="store_true",
        help="Overwrite the output directory.",
    )
    parser.add_argument(
        "--continue_training",
        action="store_true",
        help="Continue training from the last checkpoint.",
    )
    parser.add_argument(
        "--save_steps",
        type=int,
        default=-1,
        help="Save checkpoint every X updates steps (default after each epoch).",
    )
    parser.add_argument(
        "--save_total_limit",
        type=int,
        default=None,
        help="Maximum number of checkpoints to keep",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for initialization."
    )
    parser.add_argument(
        "--train_batch_size", default=8, type=int, help="Batch size for training."
    )
    parser.add_argument(
        "--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps."
    )
    parser.add_argument(
        "--weight_decay", default=0.0, type=float, help="Weight decay if we apply some."
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
        default=0.1,
        type=float,
        help="IRM penalty weight",
    )
    parser.add_argument(
        "--use_leak_probe",
        default=0,
        type=int,
        help="Use pretrained leakage probe ψ during training (0 / 1).",
    )
    parser.add_argument(
        "--psi_model_path",
        default=None,
        type=str,
        help="Path to the saved ψ .pt checkpoint (state_dict).",
    )
    parser.add_argument(
        "--leak_penalty_weight",
        default=0.3,
        type=float,
        help="λ_leak: weight for leakage penalty from ψ.",
    )

    args = parser.parse_args()

    current_path = os.path.dirname(os.path.abspath(__file__))

    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir, exist_ok=True)
        print("Directory '% s' created" % args.out_dir)

    if (
        os.path.exists(args.out_dir)
        and len(os.listdir(args.out_dir)) > 1
        and args.do_train
        and not args.overwrite_out_dir
        and not args.continue_training
    ):
        raise ValueError(
            f"Output directory {args.out_dir} already exists and is not empty. "
            f"Use --overwrite_out_dir or --continue_training."
        )

    # Setup device
    device = torch.device(
        f"cuda:{args.device}"
        if torch.cuda.is_available() and args.device != "cpu"
        else "cpu"
    )
    # Setup logging
    logging_path = os.path.join(args.out_dir, "logger.log")
    if os.path.exists(logging_path):
        os.remove(logging_path)
    handlers = [
            logging.FileHandler(logging_path),
            logging.StreamHandler(),
    ]
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
        handlers=handlers,
    )
    logger = logging.getLogger(__name__)

    # load data
    if args.data_type == 'regular':
      if args.task == 'ECQA':
        args.data_path = os.path.join(current_path, '../', 'dataset', 'ecqa', 'output')
        args.data_path = os.path.normpath(args.data_path)
        args.train_file = os.path.join(args.data_path, 'ecqa_train.csv')
        args.val_file = os.path.join(args.data_path, 'ecqa_val.csv')
        args.test_file = os.path.join(args.data_path, 'ecqa_test.csv')
      elif args.task == 'ESNLI':
        args.data_path = os.path.join(current_path, '../', 'dataset', 'esnli', 'output')
        args.data_path = os.path.normpath(args.data_path)
        args.train_file = os.path.join(args.data_path, 'esnli_train_templated.jsonl')
        args.val_file = os.path.join(args.data_path, 'esnli_val_templated.jsonl')
        args.test_file = os.path.join(args.data_path, 'esnli_test_templated.jsonl')
    # Put the constructed baseline rationales for gold labels under 'generate_baseline_rationales'
    elif args.data_type == 'temp':
      args.train_file  = os.path.join(current_path, '../', 'generate_baseline_rationales', 'output', args.task, 'baseline_rationales_train_output.jsonl')
      args.train_file = os.path.normpath(args.train_file)
      args.val_file  = os.path.join(current_path, '../', 'generate_baseline_rationales', 'output', args.task, 'baseline_rationales_val_output.jsonl')
      args.val_file = os.path.normpath(args.val_file)
      args.test_file = os.path.join(current_path, '../', 'generate_baseline_rationales', 'output', args.task, 'baseline_rationales_test_output.jsonl')
      args.test_file = os.path.normpath(args.test_file)

    if not os.path.exists(args.train_file):
        raise ValueError(f"Training file {args.train_file} not found")
    logger.info(f"Training file: {args.train_file}")
    logger.info(f"Validation file: {args.val_file}")

    #sanity check
    if args.data_type == 'temp':
        if args.use_irm == 1:
            raise ValueError("IRM training is only supported for regular model Φ. Please read the definition of Φ and Ψ in the paper.")
        if args.use_leak_probe == 1:
            raise ValueError("Leakage probing is only supported for regular model Φ. Please read the definition of Φ and Ψ in the paper.")

    # Set seed
    set_seed(args)

    # Load the models
    if args.continue_training:
        args.model_name_or_path = args.out_dir

    args.device = "cpu"
    tokenizer, model = init_model(
        args.model_name_or_path, device=args.device, do_lower_case=args.do_lower_case
    )

    args.pad_token_id = tokenizer.pad_token_id
    logger.info(f"Pad token ID: {args.pad_token_id}")
    args.block_size = tokenizer.max_len_single_sentence

    # Add special tokens
    if args.do_train and not args.continue_training:
        tokenizer.pad_token = "<pad>"
        tokenizer.eos_token = "<eos>"
        original_special_tokens = tokenizer.additional_special_tokens
        new_special_tokens = special_toks[args.task]

        all_special_tokens= list(OrderedDict.fromkeys(original_special_tokens + new_special_tokens))
        tokenizer.add_special_tokens({'additional_special_tokens': all_special_tokens})
        # logger.info(f"ALL special tokens: {all_special_tokens}")

        model.config.eos_token_id = tokenizer.convert_tokens_to_ids("<eos>")
        model.resize_token_embeddings(len(tokenizer))

    args.pad_token_id = tokenizer.pad_token_id

    args.device = device
    model.to(args.device)
    if hasattr(model, "gradient_checkpointing_enable"):
        logger.info("Enable gradient checkpointing")
        model.gradient_checkpointing_enable()

    # Training
    if args.do_train:
        args.task = transforming_task_type(args)
        train_dataset = EncoderDecoderTextDataset(
            tokenizer,
            args,
            file_path=args.train_file,
            block_size=args.block_size,
            is_train=True,
        )
        args.task = args.task.replace('_FILTERED', '').replace('_IRM', '')  # reset back to the original task name
        if args.do_eval or args.eval_during_training:
            logger.info(f"{args.task}")
            eval_dataset = EncoderDecoderTextDataset(
            tokenizer, args, file_path=args.val_file, block_size=args.block_size, is_train=False)

        # debug_print_dataloader_samples(tokenizer, train_dataset, n_samples=5, batch_size=min(8, args.train_batch_size))

        global_step, tr_loss = train(
            args,
            train_dataset,
            model,
            tokenizer,
            eval_dataset=eval_dataset,
        )

        logger.info(f" global_step = {global_step}, average loss = {tr_loss}")
        print("Training complete. Global step: {}, Average loss: {}".format(global_step, tr_loss))

    # Evaluation
    if args.do_eval:
        checkpoint = args.out_dir
        logger.info(f"Evaluate the following checkpoint: {checkpoint}")
        tokenizer, model = init_model(
            checkpoint, device=args.device, do_lower_case=args.do_lower_case
        )

        model.to(args.device)
        eval_dataset = EncoderDecoderTextDataset(
            tokenizer, args, file_path=args.val_file, block_size=args.block_size, is_train=False)

        evaluate(eval_dataset, args, model, prefix='last_step', tokenizer=tokenizer, save_predictions=False)

        logger.info("Evaluation Done")


    return


if __name__ == "__main__":
    main()
