import os
import torch
import logging
import argparse
from psi_model import Seq2SeqLeakProbe
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from torch.optim import AdamW 
from tqdm import tqdm
from collections import OrderedDict
from generative import set_seed

from utils import load_data_leak_probe
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

load_data_func = {
    'ECQA': load_data_leak_probe,
    'ESNLI': load_data_leak_probe,
}

special_toks = {
    'ECQA': ["[question]", "[choice]", "[answer]", "[rationale]", "<eos>", "<mask>"],
    'ESNLI': ["[premise]", "[hypothesis]", "[answer]", "[rationale]", "<eos>", "<mask>"],
}

class EncoderDecoderTextDataset(Dataset):
    """
    dataset for ψ:
    input: neutralized baseline (tilde_B)
    target: answer_text（token sequence）
    """
    def __init__(self, tokenizer, args, file_path):
        super().__init__()
        logger.info("Converting to token IDs")
        examples = load_data_func[args.task](file_path)

        logger.info(examples[:3])
        self.inputs = []
        self.outputs = []
        self.input_lengths = []
        self.output_lengths = []

        for idx, ex in enumerate(examples):
            source_text, target_text = ex
            
            input_enc = tokenizer(
                source_text,
                max_length=args.max_input_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt',
                add_special_tokens=True,
            )
            output_enc = tokenizer(
                target_text,
                max_length=args.max_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt',
                add_special_tokens=True,
            )

            self.inputs.append(input_enc['input_ids'].squeeze(0))
            self.outputs.append(output_enc['input_ids'].squeeze(0))
            self.input_lengths.append((input_enc['attention_mask'] == 1).sum().item())
            self.output_lengths.append((output_enc['attention_mask'] == 1).sum().item())

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

        return data

def build_phi_model(args):
    # load finetuned regular model Φ₀ as frozen encoder to extract features
    phi_model = AutoModelForSeq2SeqLM.from_pretrained(args.phi_model_path)
    logger.info(f"Loading Φ₀ (baseline QA model) from: {args.phi_model_path}")
    phi_model.to(args.device)
    phi_model.eval()
    for p in phi_model.parameters():
        p.requires_grad = False
    return phi_model

def get_encoder(model):
    if hasattr(model, "get_encoder"):
        return model.get_encoder()
    if hasattr(model, "encoder"):
        return model.encoder
    raise AttributeError("This model has no accessible encoder.")


def quick_test_from_test_dataset(tokenizer, args, test_dataset, k=10):
    """
    use few examples from test set to do a quick test
    - input: B_tilde
    - encoder: Φ₀
    - decoder: ψ
    - see what answers ψ generates compared to the true answers
    """
    logger.info("***** Quick test on test set *****")

    # Φ₀
    phi_model = build_phi_model(args)

    # ψ
    psi_model = Seq2SeqLeakProbe(args.model_hf_id, tokenizer=tokenizer)
    state_dict = torch.load(os.path.join(args.out_dir, args.out_name), map_location=args.device)
    psi_model.load_state_dict(state_dict)
    psi_model.to(args.device)
    psi_model.eval()

    num_examples = min(k, len(test_dataset))

    with torch.no_grad():
        for idx in range(num_examples):
            sample = test_dataset[idx]
            input_ids = sample["inputs"].unsqueeze(0).to(args.device)
            input_mask = sample["input_mask"].unsqueeze(0).to(args.device)
            target_ids = sample["outputs"].unsqueeze(0).to(args.device)
            output_mask = sample["output_mask"].unsqueeze(0).to(args.device)

            labels = target_ids.clone()
            labels[output_mask == 0] = -100

            # 1) Φ₀ encoder on B_tilde
            encoder = get_encoder(phi_model)
            enc_out = encoder(
                input_ids=input_ids,
                attention_mask=input_mask,
                return_dict=True,
            )
            encoder_hidden_states = enc_out.last_hidden_state  # [1, L_enc, d]

            # 2) ψ decoder to get loss and logits
            loss, logits = psi_model(
                encoder_hidden_states=encoder_hidden_states,
                labels=labels,
                encoder_attention_mask=input_mask,
            )

            # use argmax as a rough decode
            pred_ids = logits[0].argmax(dim=-1)
            input_text = tokenizer.decode(sample["inputs"], skip_special_tokens=True)
            target_text = tokenizer.decode(sample["outputs"], skip_special_tokens=True)
            pred_text = tokenizer.decode(pred_ids, skip_special_tokens=True)

            logger.info(f"\n=== Test sample {idx} ===")
            logger.info(f"Input (B_tilde): {input_text}")
            logger.info(f"Gold answer   : {target_text}")
            logger.info(f"Pred answer   : {pred_text}")
            logger.info(f"Loss          : {loss.item():.4f}")



def train_leak_probe(args, tokenizer, train_dataset, eval_dataset):

    train_dataloader = DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=True)
    val_dataloader = DataLoader(eval_dataset, batch_size=args.eval_batch_size, shuffle=False)

    # Φ₀: use regular model as feature extractor
    phi_model = build_phi_model(args)

    # ψ：decoder-only probe (the object of this training script)
    psi_model = Seq2SeqLeakProbe(args.model_hf_id, tokenizer=tokenizer).to(args.device)

    optimizer = AdamW(
        filter(lambda p: p.requires_grad, psi_model.parameters()),
        lr=args.learning_rate,
        eps=args.adam_epsilon, 
        weight_decay=args.weight_decay,
    )

    best_val_loss = float("inf")

    for epoch in range(int(args.num_train_epochs)):
        psi_model.train()
        total_loss = 0

        for batch in tqdm(train_dataloader):
            input_ids = batch["inputs"].to(args.device)
            input_mask = batch["input_mask"].to(args.device)
            target_ids = batch["outputs"].to(args.device)
            output_mask = batch["output_mask"].to(args.device)
            
            labels = target_ids.clone()
            labels[output_mask == 0] = -100 
            
            # 1) use frozen Φ₀ encoder to get encoder hidden states
            with torch.no_grad():
                encoder = get_encoder(phi_model)
                enc_out = encoder(
                    input_ids=input_ids,
                    attention_mask=input_mask,
                    return_dict=True,
                )
                encoder_hidden_states = enc_out.last_hidden_state

            # 2) ψ decoder to get loss
            loss, _ = psi_model(
                encoder_hidden_states=encoder_hidden_states,
                labels=labels,
                encoder_attention_mask=input_mask,
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        logger.info(f"[Epoch {epoch}] Train Loss: {total_loss / len(train_dataloader):.4f}")

        # Validation
        psi_model.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for batch in tqdm(val_dataloader):
                input_ids = batch["inputs"].to(args.device)
                input_mask = batch["input_mask"].to(args.device)
                target_ids = batch["outputs"].to(args.device)
                output_mask = batch["output_mask"].to(args.device)

                labels = target_ids.clone()
                labels[output_mask == 0] = -100

                encoder = get_encoder(phi_model)
                enc_out = encoder(
                    input_ids=input_ids,
                    attention_mask=input_mask,
                    return_dict=True,
                )
                encoder_hidden_states = enc_out.last_hidden_state

                val_loss, _ = psi_model(
                    encoder_hidden_states=encoder_hidden_states,
                    labels=labels,
                    encoder_attention_mask=input_mask,
                )
                total_val_loss += val_loss.item()

        avg_val_loss = total_val_loss / max(1, len(val_dataloader))
        logger.info(f"[Epoch {epoch}] Val loss: {avg_val_loss:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            os.makedirs(args.out_dir, exist_ok=True)
            save_path = os.path.join(args.out_dir, args.out_name)
            logger.info(f"  New best val loss. Saving ψ checkpoint to {save_path}")
            torch.save(psi_model.state_dict(), save_path)


if __name__ == "__main__":


    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument(
        "--out_dir",
        default=None,
        type=str,
        required=True,
        help="Out directory for checkpoints.",
    )
    parser.add_argument(
        "--out_name",
        default=None,
        type=str,
        required=True,
        help="Out file name for phi model weight",
    )
    # Other parameters
    parser.add_argument(
        "--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer."
    )
    parser.add_argument(
        "--device", default="cpu", type=str, help="GPU number or 'cpu'."
    )
    parser.add_argument(
        "--eval_batch_size", default=16, type=int, help="Batch size for evaluation."
    )
    parser.add_argument(
        "--learning_rate",
        default=3e-5,
        type=float,
        help="The initial learning rate for Adam.",
    )
    parser.add_argument(
        "--max_input_length",
        default=300,
        type=int,
        help="Maximum input event length in words.",
    )
    parser.add_argument(
        "--max_length", default=20, type=int, required=False, help="Maximum text length"
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
        default=4.0,
        type=float,
        help="Number of training epochs to perform.",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for initialization."
    )
    parser.add_argument(
        "--train_batch_size", default=16, type=int, help="Batch size for training."
    )
    parser.add_argument(
        "--weight_decay", default=0.0, type=float, help="Weight decay if we apply some."
    )
    parser.add_argument("--task", type=str, required=True, choices=["ECQA", "ESNLI"], help="Task name (ECQA or ESNLI)"
    )
    parser.add_argument(
        "--phi_model_path",
        default=None,
        type=str,
        required=True,
        help="Path to the finetuned regular model Φ₀ checkpoint.",
    )
    args = parser.parse_args()

    current_path = os.path.dirname(os.path.abspath(__file__))

    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir, exist_ok=True)
        print("Directory '% s' created" % args.out_dir)
    
    if not os.path.exists(args.phi_model_path):
        raise ValueError(f"Φ₀ model path {args.phi_model_path} not found")

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

    args.device = device

    # Set seed
    set_seed(args)

    args.model_hf_id = args.model_name_or_path
    if args.model_hf_id == "bart-large":
        args.model_hf_id = "facebook/bart-large"

    # Load the tokenizers
    tokenizer = AutoTokenizer.from_pretrained(args.model_hf_id)
    args.pad_token_id = tokenizer.pad_token_id
    logger.info(f"Pad token ID: {args.pad_token_id}")

    # Add special tokens
    tokenizer.eos_token = "<eos>"
    original_special_tokens = tokenizer.additional_special_tokens
    new_special_tokens = special_toks[args.task]

    all_special_tokens= list(OrderedDict.fromkeys(original_special_tokens + new_special_tokens))
    tokenizer.add_special_tokens({'additional_special_tokens': all_special_tokens})
    if 'gpt' in args.model_name_or_path:
        tokenizer.add_special_tokens({'pad_token': '<pad>'})
    # logger.info(f"ALL special tokens: {all_special_tokens}")

    # load data
    args.train_file  = os.path.join(current_path, '../', 'generate_baseline_rationales', 'output', args.task, 'baseline_rationales_train_output_ig_'+args.model_name_or_path+'.jsonl')
    args.train_file = os.path.normpath(args.train_file)
    args.val_file  = os.path.join(current_path, '../', 'generate_baseline_rationales', 'output', args.task, 'baseline_rationales_val_output_ig_'+args.model_name_or_path+'.jsonl')
    args.val_file = os.path.normpath(args.val_file)
    args.test_file  = os.path.join(current_path, '../', 'generate_baseline_rationales', 'output', args.task, 'baseline_rationales_test_output_ig_'+args.model_name_or_path+'.jsonl')
    args.test_file = os.path.normpath(args.test_file)

    if not os.path.exists(args.train_file):
        raise ValueError(f"Training file {args.train_file} not found")
    if not os.path.exists(args.val_file):
        raise ValueError(f"Training file {args.val_file} not found")
    if not os.path.exists(args.test_file):
        raise ValueError(f"Training file {args.test_file} not found")
    logger.info(f"Training file: {args.train_file}")
    logger.info(f"Validation file: {args.val_file}")
    logger.info(f"Test file: {args.test_file}")


    # Training
    train_dataset = EncoderDecoderTextDataset(
        tokenizer,
        args,
        file_path=args.train_file,
    )

    eval_dataset = EncoderDecoderTextDataset(
        tokenizer, 
        args, 
        file_path=args.val_file, 
    )

    test_dataset = EncoderDecoderTextDataset(
        tokenizer,
        args,
        file_path=args.test_file,
    )
    logger.info(f"***** Running training *****")
    train_leak_probe(args, tokenizer, train_dataset, eval_dataset)
    logger.info(f"***** Running testing *****")
    quick_test_from_test_dataset(tokenizer, args, test_dataset, k=10)
