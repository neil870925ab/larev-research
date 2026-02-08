from torch.nn import CrossEntropyLoss
import torch
import logging

logger = logging.getLogger(__name__)

def get_loss(args, batch, model, tokenizer, global_step, psi_model=None):
    """
    Compute this batch loss
    """
    input_ids = batch["inputs"].to(args.device)
    input_mask = batch["input_mask"].to(args.device)
    target_ids = batch["outputs"].to(args.device)
    output_mask = batch["output_mask"].to(args.device)

    # Create proper decoder_input_ids by removing the last non-padding token from each sequence
    decoder_input_ids = torch.zeros_like(target_ids[:, :-1])
    for i in range(target_ids.size(0)):
        # Find the actual sequence length (excluding padding)
        seq_len = output_mask[i].sum().item()
        if seq_len > 1:
            # Copy all tokens except the last real token
            decoder_input_ids[i, :seq_len-1] = target_ids[i, :seq_len-1]
            # Pad the rest
            decoder_input_ids[i, seq_len-1:] = args.pad_token_id
        else:
            # If sequence is too short, just use padding
            decoder_input_ids[i, :] = args.pad_token_id

    # We don't send labels to model.forward because we want to compute per token loss
    lm_logits = model(
        input_ids, attention_mask=input_mask, decoder_input_ids=decoder_input_ids, use_cache=False
    )[0]
    batch_size, max_length, vocab_size = lm_logits.shape

    # Compute loss for each instance and each token
    loss_fct = CrossEntropyLoss(reduction="none")
    lm_labels = target_ids[:, 1:].clone().contiguous()
    lm_labels[target_ids[:, 1:] == args.pad_token_id] = -100
    loss = loss_fct(lm_logits.view(-1, vocab_size), lm_labels.view(-1)).view(
        batch_size, max_length
    )

    # Debug: Print the first sample in the batch. Only print the non-padding part
    # print("=== Debug: FIRST SAMPLE ===")
    # encoder_tokens = input_ids[0][input_mask[0] == 1]
    # print("[Encoder input / question + rationale + baseline]")
    # print(tokenizer.decode(encoder_tokens, skip_special_tokens=False))
    # decoder_tokens = decoder_input_ids[0][decoder_input_ids[0] != args.pad_token_id]
    # print("\n[Decoder input]")
    # print(tokenizer.decode(decoder_tokens, skip_special_tokens=False))
    # label_tokens = lm_labels[0][lm_labels[0] != -100]
    # print("\n[Decoder labels / answer]")
    # print(tokenizer.decode(label_tokens, skip_special_tokens=False))
    # raise ValueError("Debug stop")

    # Only consider non padded tokens
    loss_mask = output_mask[..., :-1].contiguous()
    ermloss = torch.mul(loss_mask, loss)  # [batch_size, max_length]
    erm_mean = ermloss.sum() / loss_mask.sum()

    if args.use_leak_probe == 1 and psi_model is not None and global_step is not None:
        leak_lambda = min(
            args.leak_penalty_weight * (global_step / args.leak_penalty_anneal_iters),
            args.leak_penalty_weight
        )
        leak_loss = compute_leak_loss(
            args,
            batch,
            model=model,
            psi_model=psi_model,
        )

    else:
        leak_loss = torch.tensor(0.0, device=args.device)
        leak_lambda = 0.0

    total_loss = erm_mean - (leak_lambda*leak_loss) # the lower the leak loss, more leakage the model has, so we subtract it 
    
    return total_loss, erm_mean, leak_loss


def get_irm_loss(args, batch, model, tokenizer, global_step, psi_model=None):
    """
    IRMv1 (from the paper of Arjovsky et al. 2019, Sec. 3.1).
    """

    if args.data_type != 'regular':
        raise ValueError("IRM loss is only applicable for 'regular' data_type.")

    irm_lambda = min(
        args.irm_penalty_weight * (global_step / args.irm_penalty_anneal_iters),
        args.irm_penalty_weight
    )

    input_ids = batch["inputs"].to(args.device)
    input_mask = batch["input_mask"].to(args.device)
    target_ids = batch["outputs"].to(args.device)
    output_mask = batch["output_mask"].to(args.device)
    env_ids = batch["env_ids"].to(args.device)

    # Create proper decoder_input_ids by removing the last non-padding token from each sequence
    decoder_input_ids = torch.zeros_like(target_ids[:, :-1])
    for i in range(target_ids.size(0)):
        # Find the actual sequence length (excluding padding)
        seq_len = output_mask[i].sum().item()
        if seq_len > 1:
            # Copy all tokens except the last real token
            decoder_input_ids[i, :seq_len-1] = target_ids[i, :seq_len-1]
            # Pad the rest
            decoder_input_ids[i, seq_len-1:] = args.pad_token_id
        else:
            # If sequence is too short, just use padding
            decoder_input_ids[i, :] = args.pad_token_id

    unique_envs = torch.unique(env_ids)
    erm_losses = []
    irm_penalties = []

    for e in unique_envs:
        e_mask = (env_ids == e)
        if e_mask.sum() == 0:
            continue

        # ========== Subset for this environment ==========
        input_ids_e = input_ids[e_mask]
        input_mask_e = input_mask[e_mask]
        target_ids_e = target_ids[e_mask]
        output_mask_e = output_mask[e_mask]
        decoder_input_ids_e = decoder_input_ids[e_mask]

        # ========== Forward pass ==========
        lm_logits_e = model(
            input_ids_e,
            attention_mask=input_mask_e,
            decoder_input_ids=decoder_input_ids_e,
            use_cache=False
        )[0]

        batch_size_e, max_length, vocab_size = lm_logits_e.shape
        lm_labels_e = target_ids_e[:, 1:].clone().contiguous()
        lm_labels_e[target_ids_e[:, 1:] == args.pad_token_id] = -100
        loss_mask_e = output_mask_e[..., :-1].contiguous()

        # Debug: Print the first sample in the batch. Only print the non-padding part
        # print("=== Debug: FIRST SAMPLE ===")
        # encoder_tokens = input_ids_e[0][input_mask_e[0] == 1]
        # print("[Encoder input / question + rationale + baseline]")
        # print(tokenizer.decode(encoder_tokens, skip_special_tokens=False))
        # decoder_tokens = decoder_input_ids_e[0][decoder_input_ids_e[0] != args.pad_token_id]
        # print("\n[Decoder input]")
        # print(tokenizer.decode(decoder_tokens, skip_special_tokens=False))
        # label_tokens = lm_labels_e[0][lm_labels_e[0] != -100]
        # print("\n[Decoder labels / answer]")
        # print(tokenizer.decode(label_tokens, skip_special_tokens=False))

        # raise ValueError("Debug stop")



        loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
        loss_e = loss_fct(lm_logits_e.view(-1, vocab_size), lm_labels_e.view(-1)).view(
            batch_size_e, max_length
        )
        loss_e = (loss_e * loss_mask_e).sum() / loss_mask_e.sum()
        erm_losses.append(loss_e)

        # ========== IRM penalty (first-order surrogate) ==========
        scale = torch.tensor(1.0, device=args.device, requires_grad=True)
        scaled_logits_e = lm_logits_e * scale
        irm_loss_e = loss_fct(scaled_logits_e.view(-1, vocab_size),
                              lm_labels_e.view(-1)).view(batch_size_e, max_length)
        irm_loss_e = (irm_loss_e * loss_mask_e).sum() / loss_mask_e.sum()

        # ∇_scale L : compute gradient w.r.t. scale
        grad = torch.autograd.grad(irm_loss_e, [scale], create_graph=True, retain_graph=True)[0]
        irm_penalty = grad.pow(2).sum()
        irm_penalties.append(irm_penalty)

    if len(erm_losses) == 0:
        raise ValueError("No environments found in the batch for IRM loss computation.")
    
    if args.use_leak_probe == 1 and psi_model is not None and global_step is not None:
        leak_lambda = min(
            args.leak_penalty_weight * (global_step / args.leak_penalty_anneal_iters),
            args.leak_penalty_weight
        )
        leak_loss = compute_leak_loss(
            args,
            batch,
            model=model,
            psi_model=psi_model,
        )

    else:
        leak_loss = torch.tensor(0.0, device=args.device)
        leak_lambda = 0.0

    erm_mean = torch.stack(erm_losses).mean()
    irm_penalty_mean = torch.stack(irm_penalties).mean()

    total_loss = erm_mean + (irm_lambda*irm_penalty_mean) - (leak_lambda*leak_loss) # the lower the leak loss, more leakage the model has, so we subtract it 
    return total_loss, erm_mean, irm_penalty_mean, leak_loss

def compute_leak_loss(args, batch, model, psi_model):
    """
    use B_tilde (batch['inputs_leak']) to run Φ's encoder,
    then feed encoder_hidden_states to ψ to compute leak loss.
    """
    if args.data_type != 'regular':
        raise ValueError("Leak Probe loss is only applicable for 'regular' data_type.")
    
    input_ids_leak = batch["inputs_leak"].to(args.device)          # [B, L_enc]
    input_leak_mask = batch["input_leak_mask"].to(args.device)     # [B, L_enc]

    target_ids  = batch["outputs"].to(args.device)                 # [B, L_dec]
    output_mask = batch["output_mask"].to(args.device)             # [B, L_dec]

    labels = target_ids.clone()
    labels[output_mask == 0] = -100

    # use Φ's encoder to encode B_tilde to get encoder_hidden_states
    encoder_module = model.get_encoder() if hasattr(model, "get_encoder") else model.encoder
    if encoder_module is None:
        # some models only have get_encoder() method
        encoder_module = model.get_encoder()

    encoder_outputs = encoder_module(
        input_ids=input_ids_leak,
        attention_mask=input_leak_mask,
        return_dict=True,
    )
    encoder_hidden_states = encoder_outputs.last_hidden_state      # [B, L_enc, d]

    # Feed to ψ, compute cross-entropy (or ψ-defined loss)
    leak_loss, _ = psi_model(
        encoder_hidden_states=encoder_hidden_states,
        labels=labels,
        encoder_attention_mask=input_leak_mask
    )

    return leak_loss


