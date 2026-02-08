# Adapted from:
# https://github.com/HanjieChen/REV/blob/main/src/rev/generative.py
"""
Generative model to predict intensifiers and attenuators given an optional premise and a hypothesis.
Based on https://github.com/huggingface/transformers/blob/master/examples/run_lm_finetuning.py:
fine-tuning language models on a text file using a causal language modeling (CLM) loss.
"""
import os
import torch
import random
import logging
import numpy as np
import json
import hashlib

from tqdm import tqdm, trange

from torch.optim import AdamW
from transformers.optimization import get_linear_schedule_with_warmup

from torch.utils.data import DataLoader, Sampler

from utils import load_data_ecqa, load_data_esnli, load_data_ecqa_irm, load_data_esnli_irm
from loss_fnc import get_loss, get_irm_loss
from psi_model import Seq2SeqLeakProbe

logger = logging.getLogger(__name__)

load_data_func = {
    'ECQA': load_data_ecqa,
    'ESNLI': load_data_esnli,
    'ECQA_IRM': load_data_ecqa_irm,
    'ESNLI_IRM': load_data_esnli_irm,
}

def normalize_pred_for_bart(pred):
    output = pred.replace("[answer]<s>", "[answer]").replace("<s>", "").strip()
    return output

def transforming_task_type(args):
    if args.use_irm == 1:
        return args.task + '_IRM'
    else:
        return args.task

def create_dataloader(dataset, args, batch_size=None, is_train=True):
    """
    Create a DataLoader for either:
      (1) Normal training (standard DataLoader) / Evaluation for IRM training / Evaluation for standard training
      (2) IRM training (grouped by q_id, each batch = 1 question × all environments)
    """

    def create_standard_dataloader():
        def collate_fn(batch):
            result = {}
            for key in batch[0].keys():
                if isinstance(batch[0][key], torch.Tensor):
                    result[key] = torch.stack([d[key] for d in batch])
                else:
                    result[key] = [d[key] for d in batch]
            return result
        
        g = torch.Generator()
        g.manual_seed(args.seed)
        
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            generator=g,
            collate_fn=collate_fn
        )

    def create_irm_dataloader():
        class QuestionBatchSampler(Sampler):
            def __init__(self, qid_list, num_envs):
                self.qid_list = qid_list
                self.unique_qids = sorted(set(qid_list))
                self.num_envs = num_envs

            def __iter__(self):
                random.shuffle(self.unique_qids)
                for q in self.unique_qids:
                    indices = [i for i, qid in enumerate(self.qid_list) if qid == q]
                    if len(indices) == self.num_envs:
                        yield indices
                    else:
                        raise ValueError(f"Question ID {q} only has {len(indices)} / {self.num_envs} environments. Skipping.")

            def __len__(self):
                return len(self.unique_qids)

        q_ids = dataset.question_ids
        env_ids = dataset.env_ids
        num_envs = len(set(env_ids))

        if batch_size != num_envs:
            raise ValueError(f"For IRM training, batch_size must be equal to the number of environments ({num_envs}).")

        sampler = QuestionBatchSampler(qid_list=q_ids, num_envs=num_envs)

        def collate_fn(batch):
            result = {}
            for key in batch[0].keys():
                if isinstance(batch[0][key], torch.Tensor):
                    result[key] = torch.stack([b[key] for b in batch])
                else:
                    result[key] = [b[key] for b in batch]
            return result

        dataloader = DataLoader(
            dataset,
            batch_sampler=sampler,
            collate_fn=collate_fn,
        )

        return dataloader

    if args.use_irm == 1 and is_train:
        return create_irm_dataloader()
    else:
        return create_standard_dataloader()

def log_train_step(args, global_step, erm_mean, irm_penalty_mean, batch_loss, leak_loss):
    parts = [f"[step {global_step}]"]

    # ERM
    if erm_mean is not None:
        parts.append(f"---ERM={erm_mean:.4f}")

    # IRM
    if args.use_irm == 1 and irm_penalty_mean is not None:
        current_irm_lambda = min(
            args.irm_penalty_weight * (global_step / args.irm_penalty_anneal_iters),
            args.irm_penalty_weight
        )
        parts.append(f"---IRM_penalty={irm_penalty_mean:.4f}")
        parts.append(f"---lambda_IRM={current_irm_lambda:.4f}")

    # Leak probe
    if args.use_leak_probe == 1:
        current_leak_lambda = min(
            args.leak_penalty_weight * (global_step / args.leak_penalty_anneal_iters),
            args.leak_penalty_weight
        )
        parts.append(f"---Leak={leak_loss:.4f}")
        parts.append(f"---lambda_Leak={current_leak_lambda:.4f}")

    # Total loss
    parts.append(f"---Total={batch_loss:.4f}")

    logger.info(" | ".join(parts))


def main():
    return None

def set_seed(args):
    """
    Set the random seed for reproducibility
    """
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False

def check_irm_dataloader_integrity(dataloader, expected_envs=3):
    """
    Check that each batch in the IRM dataloader contains exactly one question ID
    and the expected number of unique environment IDs.
    Returns True if all batches are valid, False otherwise.
    """
    import torch
    from collections import Counter

    total_batches = 0
    error_batches = 0
    q_error_count = 0

    for batch_idx, batch in enumerate(dataloader):
        q_ids = batch["question_ids"]
        env_ids = batch["env_ids"]

        if torch.is_tensor(q_ids): q_ids = q_ids.tolist()
        if torch.is_tensor(env_ids): env_ids = env_ids.tolist()

        unique_q = set(q_ids)
        if len(unique_q) != 1:
            logger.warning(f"Batch {batch_idx}: contains multiple q_id → {unique_q}")
            error_batches += 1
            continue

        qid = list(unique_q)[0]
        env_count = Counter(env_ids)

        if len(env_count) != expected_envs:
            logger.warning(f"Batch {batch_idx}, q_id={qid}: found {len(env_count)} environments ({list(env_count.keys())})")
            error_batches += 1
            q_error_count += 1
        elif any(v > 1 for v in env_count.values()):
            logger.warning(f"Batch {batch_idx}, q_id={qid}: contains duplicate env_id {env_count}")
            error_batches += 1
            q_error_count += 1

        total_batches += 1

    if error_batches == 0:
        return True
    else:
        return False


def train(args, train_dataset, model, tokenizer, eval_dataset=None):
    """
    Train the model.
    """
    set_seed(args)
    train_dataloader = create_dataloader(train_dataset, args, args.train_batch_size, is_train=True)

    if args.use_irm == 1:
        is_clean = check_irm_dataloader_integrity(train_dataloader, expected_envs=3)
        if not is_clean:
            raise ValueError("IRM dataloader integrity for train_dataloader check failed. Please fix the dataset or dataloader.")
        logger.info("Using IRM training")

    psi_model = None
    if args.use_leak_probe == 1:
        if args.psi_model_path is None:
            raise ValueError("Leak probe model path must be specified when use_leak_probe is set to 1.")
        if args.leak_penalty_weight <= 0:
            raise ValueError("Leak probe penalty weight must be positive when use_leak_probe is set to 1.")
        
        logger.info(f"Loading leakage probe ψ from {args.psi_model_path}")
        psi_model = Seq2SeqLeakProbe(model_name_or_path=args.model_name_or_path, tokenizer=tokenizer)
        set_seed(args)
        state_dict = torch.load(args.psi_model_path, map_location=args.device)
        psi_model.load_state_dict(state_dict)
        psi_model.to(args.device)
        psi_model.eval()
        for p in psi_model.parameters():
            p.requires_grad = False
        logger.info("Leak probe ψ loaded and frozen.")


    # Set the number of steps based on the num_epochs * len(train) or args.max_steps if specified.
    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = (
            args.max_steps
            // (len(train_dataloader) // args.gradient_accumulation_steps)
            + 1
        )
    else:
        t_total = (
            len(train_dataloader)
            // args.gradient_accumulation_steps
            * args.num_train_epochs
        )

    args.irm_penalty_anneal_iters = int(t_total / 3)
    args.leak_penalty_anneal_iters = int(t_total / 3)

    # Prepare optimizer and scheduler (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]

    optimizer = AdamW(
        optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon
    )
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
    )

    # Check if saved optimizer or scheduler states exist and load from there
    if os.path.isfile(
        os.path.join(args.model_name_or_path, "optimizer.pt")
    ) and os.path.isfile(os.path.join(args.model_name_or_path, "scheduler.pt")):
        optimizer.load_state_dict(
            torch.load(os.path.join(args.model_name_or_path, "optimizer.pt"))
        )
        scheduler.load_state_dict(
            torch.load(os.path.join(args.model_name_or_path, "scheduler.pt"))
        )

    # Train
    total_batch_size = args.train_batch_size * args.gradient_accumulation_steps
    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  learning rate = {args.learning_rate}")
    logger.info(f"  Instantaneous batch size = {args.train_batch_size}")
    logger.info(
        f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}"
    )
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {t_total}")

    if args.use_irm == 1:
        logger.info(f"  IRMv1 mode active (λ={args.irm_penalty_weight}, envs=3)")
    if args.use_leak_probe == 1:
        logger.info(f"  Leak probe mode active (λ={args.leak_penalty_weight})")

    global_step, epochs_trained, steps_trained_in_current_epoch = 0, 0, 0
    tr_loss, logging_loss, eval_best_loss = 0.0, 0.0, float('inf')
    early_stop, early_stop_thr = 0, 8

    set_seed(args)
    model_to_resize = model.module if hasattr(model, "module") else model
    model_to_resize.resize_token_embeddings(len(tokenizer))

    set_seed(args)
    train_iterator = trange(epochs_trained, int(args.num_train_epochs), desc="Epoch")

    if args.use_irm == 1 and args.gradient_accumulation_steps != 1:
        raise ValueError("IRM training requires gradient_accumulation_steps = 1.")

    for _ in train_iterator:
        if early_stop > early_stop_thr:
            break
        model.zero_grad()
        epoch_iterator = tqdm(train_dataloader, desc="Iteration")
        for step, batch in enumerate(epoch_iterator):
            # Skip trained steps (resume)
            if steps_trained_in_current_epoch > 0:
                steps_trained_in_current_epoch -= 1
                continue

            model.train()

            # ==========================
            # Forward & Loss
            # ==========================
            if args.use_irm == 1:
                batch_loss, erm_mean, irm_penalty_mean, leak_loss = get_irm_loss(args, batch, model, tokenizer, global_step, psi_model=psi_model)
            else:
                batch_loss, erm_mean, leak_loss = get_loss(args, batch, model, tokenizer, global_step, psi_model=psi_model)

            loss = batch_loss

            # ==========================
            # IRM MODE
            # ==========================
            if args.use_irm == 1:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()
                tr_loss += loss.item()
                global_step += 1
                model.zero_grad()


            # ==========================
            # STANDARD REV MODE
            # ==========================
            else:
                loss = loss / args.gradient_accumulation_steps
                loss.backward()
                tr_loss += loss.item()

                if (step + 1) % args.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                    optimizer.step()
                    scheduler.step()
                    global_step += 1
                    model.zero_grad()

            # ==========================
            # LOGGING & CHECKPOINT
            # ==========================
            if (step + 1) % args.gradient_accumulation_steps == 0:
                if args.logging_steps > 0 and global_step % args.logging_steps == 0 and global_step != 0:
                    logger.info(f"***** Train loss @ step {global_step}: {tr_loss / max(1, global_step):.4f} *****")

                    log_train_step(
                        args=args,
                        global_step=global_step,
                        erm_mean=erm_mean,
                        irm_penalty_mean=irm_penalty_mean if args.use_irm == 1 else None,
                        batch_loss=batch_loss,
                        leak_loss=leak_loss,
                    )

                    if args.eval_during_train:
                        current_eval_loss = evaluate(eval_dataset, args, model, prefix="routine", tokenizer=tokenizer, save_predictions=False)
                        if current_eval_loss < eval_best_loss:
                            eval_best_loss = current_eval_loss
                            early_stop = 0
                            if not os.path.exists(args.out_dir):
                                os.makedirs(args.out_dir)

                            logger.info(f"Saving best model checkpoint to {args.out_dir}")
                            model_to_save = model.module if hasattr(model, "module") else model
                            model_to_save.save_pretrained(args.out_dir)
                            tokenizer.save_pretrained(args.out_dir)
                            torch.save(args, os.path.join(args.out_dir, "training_args.bin"))

                            if args.save_eval_predictions_during_train:
                                logger.info("Found better model, saving predictions...")
                                _ = evaluate(eval_dataset, args, model, prefix=f"{global_step}", tokenizer=tokenizer, save_predictions=True)
                        else:
                            early_stop += 1

                if args.save_steps > 0 and global_step % args.save_steps == 0:
                    checkpoint_prefix = "checkpoint"
                    out_dir = os.path.join(args.out_dir, f"{checkpoint_prefix}-{global_step}")
                    os.makedirs(out_dir, exist_ok=True)

                    model_to_save = model.module if hasattr(model, "module") else model
                    model_to_save.save_pretrained(out_dir)
                    tokenizer.save_pretrained(out_dir)
                    torch.save(args, os.path.join(out_dir, "training_args.bin"))
                    torch.save(optimizer.state_dict(), os.path.join(out_dir, "optimizer.pt"))
                    torch.save(scheduler.state_dict(), os.path.join(out_dir, "scheduler.pt"))
                    logger.info(f"Checkpoint saved to {out_dir}")

            if 0 < args.max_steps < global_step:
                epoch_iterator.close()
                break

        if 0 < args.max_steps < global_step:
            train_iterator.close()
            break

    return global_step, tr_loss / max(1, global_step)


def evaluate(eval_dataset, args, model, prefix="", tokenizer=None, save_predictions=False):
    """
    Evaluation
    """
    eval_out_dir = args.out_dir

    if not os.path.exists(eval_out_dir):
        os.makedirs(eval_out_dir)

    eval_dataloader = create_dataloader(eval_dataset, args, args.eval_batch_size, is_train=False)
    eval_loss = None

    if tokenizer is None:
        raise ValueError("Tokenizer must be provided for evaluation.")
    else:
        if not save_predictions:

            logger.info(f"***** Running evaluation {prefix} *****")
            logger.info(f"  Num examples = {len(eval_dataset)}")
            logger.info(f"  Batch size = {args.eval_batch_size}")
            macro_loss = 0.0
            num_batches = 0
            model.eval()
            

            for batch in tqdm(eval_dataloader, desc="Evaluating"):

                with torch.no_grad():
                    # For evaluation, we don't need IRM or Leak Loss, just standard loss

                    batch_loss, _, _ = get_loss(args, batch, model, tokenizer, global_step=None, psi_model=None)
                    macro_loss += batch_loss.item()
                                
                num_batches += 1
            eval_loss = torch.tensor(macro_loss / num_batches)


            logger.info(f"***** eval_loss: {eval_loss} *****")
        
        else:
            try:
                examples = load_data_func[args.task](args, args.val_file, data_type=args.data_type, shuffle=False) #no matter irm or not, we load standard val_file

                predictions = []
                
                for source_text, target_text, leak_source_text in tqdm(examples, desc="Generating predictions"):
                    try:
                        preds = generate_conditional_eval(
                            tokenizer,
                            model,
                            args,
                            source_text,
                            target_text,
                            args.device,
                        )
                    except Exception as exp:
                        logger.info(f"Generation error: {exp}")
                        preds = []
                    
                    prediction_data = {
                        "input": source_text,
                        "label": target_text,
                        "predictions": preds
                    }
                    predictions.append(prediction_data)
                    
            except Exception as e:
                logger.warning(f"Could not load original data for prediction generation: {e}")
                logger.warning("Prediction generation will be skipped")


            predictions_file = os.path.join(eval_out_dir, f"{prefix}_b4computevi_predictions.jsonl" if prefix else "b4computevi_predictions.jsonl")
            with open(predictions_file, "w", encoding="utf-8") as f:
                for pred in predictions:
                    f.write(json.dumps(pred, ensure_ascii=False) + "\n")
            logger.info(f"Saved {len(predictions)} predictions to {predictions_file}")

    return eval_loss


def generate_conditional_eval(tokenizer, model, args, input, output, device):
    """
    Generate a sequence with models like Bart and T5 (for evaluation)
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
    
    return preds

if __name__ == "__main__":
    main()
