import torch.nn as nn
from transformers import AutoModelForSeq2SeqLM
from transformers.modeling_outputs import BaseModelOutput


class Seq2SeqLeakProbe(nn.Module):
    """
    ψ model: frozen encoder + trainable decoder (probe).
    Input: encoder_hidden_states from Φ encoder on B_tilde
    Output: answer_text loss/logits (teacher forcing via labels)
    """

    def __init__(self, model_name_or_path="t5-large", tokenizer=None):
        super().__init__()

        model_hf_id = model_name_or_path
        if model_hf_id == "bart-large":
            model_hf_id = "facebook/bart-large"
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_hf_id)

        # Resize token embeddings if you added special tokens during training
        if tokenizer is not None:
            if len(tokenizer) != self.model.get_input_embeddings().num_embeddings:
                self.model.resize_token_embeddings(len(tokenizer))

            try:
                eos_id = tokenizer.convert_tokens_to_ids("<eos>")
                if eos_id is not None and eos_id != tokenizer.unk_token_id:
                    self.model.config.eos_token_id = eos_id
            except Exception:
                pass


        self.model_type = getattr(self.model.config, "model_type", "").lower()

        # Apply the "frozen encoder + trainable decoder" policy
        if self.model_type == "t5":
            self._configure_t5_probe()
        elif self.model_type == "bart":
            self._configure_bart_probe()

        # sanity check: encoder must be frozen
        enc_params = list(self.model.get_encoder().parameters()) if hasattr(self.model, "get_encoder") else []
        enc_trainable = sum(int(p.requires_grad) for p in enc_params)
        if enc_trainable != 0:
            raise ValueError("[Seq2SeqLeakProbe] Encoder parameters should be frozen, but some are trainable!")

    def _configure_t5_probe(self):
        """
        T5 has shared embedding weights used by both encoder & decoder and tied lm_head.
        If we freeze encoder/shared, decoder embeddings + lm_head would also be frozen.
        So we:
          1) keep encoder shared embedding (frozen)
          2) create a NEW decoder embedding initialized from shared
          3) re-tie lm_head to the NEW decoder embedding
          4) freeze encoder + original shared
          5) train decoder + new decoder embedding + lm_head
        """
        # T5 internals: model has .shared, .encoder, .decoder, .lm_head
        if not (hasattr(self.model, "shared") and hasattr(self.model, "encoder") and hasattr(self.model, "decoder")):
            raise ValueError("[Seq2SeqLeakProbe] Unexpected T5 structure; cannot apply T5 probe config.")

        encoder_shared = self.model.shared  # keep reference to freeze

        vocab_size = self.model.config.vocab_size
        d_model = getattr(self.model.config, "d_model", None)
        if d_model is None:
            # fallback: infer from shared embedding
            d_model = encoder_shared.weight.shape[1]

        # Create NEW decoder embedding
        decoder_embed = nn.Embedding(vocab_size, d_model)
        decoder_embed.weight = nn.Parameter(encoder_shared.weight.clone())

        # Plug into decoder
        self.model.decoder.set_input_embeddings(decoder_embed)

        # Re-link lm_head to NEW decoder embedding
        if hasattr(self.model, "lm_head") and self.model.lm_head is not None:
            self.model.lm_head.weight = decoder_embed.weight

        # Freeze encoder + original shared
        for p in self.model.encoder.parameters():
            p.requires_grad = False
        for p in encoder_shared.parameters():
            p.requires_grad = False

        # Train decoder + new decoder embedding + lm_head
        for p in self.model.decoder.parameters():
            p.requires_grad = True
        for p in decoder_embed.parameters():
            p.requires_grad = True
        if hasattr(self.model, "lm_head") and self.model.lm_head is not None:
            for p in self.model.lm_head.parameters():
                p.requires_grad = True

    def _configure_bart_probe(self):
        """
        BART has shared embedding weights used by both encoder & decoder and tied lm_head.
        If we freeze encoder/shared, decoder embeddings + lm_head would also be frozen.
        So we:
            1) keep encoder shared embedding (frozen)
            2) create a NEW decoder embedding initialized from shared
            3) re-tie lm_head to the NEW decoder embedding
            4) freeze encoder + original shared
            5) train decoder + new decoder embedding + lm_head
        """

        # --- basic structure checks ---
        if not (hasattr(self.model, "model") and hasattr(self.model.model, "encoder") and hasattr(self.model.model, "decoder")):
            raise ValueError("[Seq2SeqLeakProbe] Unexpected BART structure; cannot untie embeddings.")

        bart = self.model.model  # BartModel
        encoder = bart.encoder
        decoder = bart.decoder

        if not hasattr(bart, "shared"):
            raise ValueError("[Seq2SeqLeakProbe] BART has no .shared embeddings; cannot untie cleanly.")

        shared = bart.shared  # nn.Embedding(vocab, d_model)

        # Create NEW decoder embedding
        vocab_size, d_model = shared.weight.shape
        decoder_embed = nn.Embedding(vocab_size, d_model)
        decoder_embed.weight = nn.Parameter(shared.weight.detach().clone())

        # Plug into decoder
        if not hasattr(decoder, "embed_tokens"):
            raise ValueError("[Seq2SeqLeakProbe] BART decoder has no embed_tokens; cannot untie embeddings.")
        decoder.set_input_embeddings(decoder_embed)

        # Re-link lm_head to NEW decoder embedding
        if hasattr(self.model, "lm_head") and self.model.lm_head is not None:
            self.model.lm_head.weight = decoder_embed.weight

        # Freeze encoder + original shared
        for p in encoder.parameters():
            p.requires_grad = False
        for p in shared.parameters():
            p.requires_grad = False

        # Train decoder + new decoder embedding + lm_head
        for p in decoder.parameters():
            p.requires_grad = True
        for p in decoder_embed.parameters():
            p.requires_grad = True
        if hasattr(self.model, "lm_head") and self.model.lm_head is not None:
            for p in self.model.lm_head.parameters():
                p.requires_grad = True


    def forward(self, encoder_hidden_states, labels, encoder_attention_mask=None):
        """
        encoder_hidden_states: [B, L_enc, d], from Φ encoder
        labels: [B, T] (teacher forcing labels)
        encoder_attention_mask: [B, L_enc] optional
        """
        encoder_outputs = BaseModelOutput(last_hidden_state=encoder_hidden_states)

        # For seq2seq models, the encoder attention mask is passed as attention_mask
        output = self.model(
            encoder_outputs=encoder_outputs,
            attention_mask=encoder_attention_mask,
            labels=labels,
            return_dict=True,
        )
        return output.loss, output.logits
