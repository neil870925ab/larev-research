# LAREV Experimental Pipeline

This repository implements the experimental pipeline described in
**Algorithm A.1: LAREV Experimental Pipeline**, which is described in the thesis.

## Environment Setup

This project uses a Conda environment specified in `larev_environment.yml`.

Please create and activate the environment before running any commands below:

```bash
conda env create -f larev_environment.yml
conda activate larev
```

---

## Conventions

- `path/to/larev-research` denotes the root directory of this repository.
- All commands below are executed from the project root unless otherwise specified.

---

## Experiment Configuration

Before running the commands below, set:
- ${TASK}: ECQA or ESNLI
- ${BACKBONE_MODEL}: t5-large or bart-large
- ${DEVICE}: GPU index (e.g., 0)
- ${EPOCH}: number of training epochs (e.g., 8, 16)
- ${LR}: learning rate (e.g., 3e-5)
- ${IRM_PENALTY}: irm penalty weight (task-dependent, e.g., 0.1, 2, 5)
- ${LEAK_PROBE_PENALTY}: leak probe penalty weight (task-dependent, e.g., 0.1, 0.001, 0.0005)

---

## Stage I: Data Preparation

### (1) Prepare dataset splits

#### ECQA dataset

```bash
cd path/to/larev-research/dataset/ecqa
git clone https://github.com/dair-iitd/ECQA-Dataset.git
cd path/to/larev-research/dataset/cqa
bash ./run_convert_cqa_from_huggingface_to_jsonl.sh
cd path/to/larev-research/dataset/ecqa
bash ./run_generate_data.sh
```

#### e-SNLI dataset

```bash
cd path/to/larev-research/dataset/esnli
curl -L -o esnli_train.csv \
  https://raw.githubusercontent.com/OanaMariaCamburu/e-SNLI/master/dataset/esnli_train_1.csv
curl -L -o esnli_val.csv \
  https://raw.githubusercontent.com/OanaMariaCamburu/e-SNLI/master/dataset/esnli_dev.csv
curl -L -o esnli_test.csv \
  https://raw.githubusercontent.com/OanaMariaCamburu/e-SNLI/master/dataset/esnli_test.csv
```

### (2) Construct baseline rationales $b$

We use 'question_statement_text' to denote baseline input $b$

#### ECQA dataset

##### Download and extract pretrained model (ECQA baseline rationale generator)

Manually download the pretrained model ([link](https://drive.usercontent.google.com/download?id=1Il5_BHnW9Rf0T72KF2JHBIA87Lzu-F7i)) archive and place it at:

path/to/larev-research/generate_baseline_rationales/prepare_baseline_rationale/ECQA/seq2seq_converter/model_data/

Then extract the archive and construct baseline rationales using the following commands:

```bash
cd path/to/larev-research/generate_baseline_rationales/prepare_baseline_rationale/ECQA/seq2seq_converter/model_data
mkdir -p question_converter-t5-3b
tar -xzf question_converter-t5-3b.tar.gz -C question_converter-t5-3b
cd ../
bash ./run_construct_baseline_rationales.sh
```

#### e-SNLI dataset

```bash
cd path/to/larev-research/generate_baseline_rationales/prepare_baseline_rationale/ENSLI
bash ./run_construct_baseline_rationales.sh
```

### (3) Construct evaluation rationale variants $\tilde{r}$

```bash
cd path/to/larev-research/generate_data_for_ranking_metric
bash ./run_generate_data_for_ranking_metric.sh ${TASK} 0
```

---

## Stage II: Model training

We use the `data_type` argument to control which evaluator is trained:

- `data_type == temp`: trains $\Phi_{\text{base}}$ (Stage II) using baseline inputs only
- `data_type == regular`: trains $\Phi$ (Stage II) or $\Phi_{\text{LA}}$ (Stage III) using gold rationales and baseline inputs

```bash
cd path/to/larev-research/LAREV
```

### (4) Train baseline model $\Phi_{\text{base}}$

```bash
bash ./train.sh ${DEVICE} temp ${BACKBONE_MODEL} ${TASK} ${EPOCH} ${LR} 0 0
```

### (5) Train regular model $\Phi$

```bash
bash ./train.sh ${DEVICE} regular ${BACKBONE_MODEL} ${TASK} ${EPOCH} ${LR} 0 0
```

---

## Stage III: Leakage-aware refinement

### (6) Compute integrated gradients on $\Phi_{\text{base}}$

```bash
bash ./compute_ig.sh ${DEVICE} ${BACKBONE_MODEL} ${TASK}
```

### (7) Generate multiple environments

```bash
cd path/to/larev-research/generate_multi_environments
export GEMINI_API_KEY="your_gemini_api_key_here"
bash ./run_generate_multi_environments.sh ${BACKBONE_MODEL} ${TASK}
```

### (8) Train leakage probe model $\psi$

```bash
cd path/to/larev-research/LAREV
bash ./train_leak_probe_model_psi.sh ${DEVICE} ${BACKBONE_MODEL} ${TASK} ${EPOCH} ${LR}
```

### (9) Train leakage-aware evaluator $\Phi_{\text{LA}}$

```bash
bash ./train_irm.sh ${DEVICE} regular ${BACKBONE_MODEL} ${TASK} ${EPOCH} ${LR} ${IRM_PENALTY} 1 ${LEAK_PROBE_PENALTY}
```

---

## Stage IV: Evaluation

### (10) Score rationale variants

```bash
bash ./evaluate_ranking_data.sh ${DEVICE} ${BACKBONE_MODEL} ${TASK} 1 ${IRM_PENALTY} 1 ${LEAK_PROBE_PENALTY} 0
```

### Optional: Score rationale variants from task-model-generated rationales

```bash
cd path/to/larev-research/generate_rationale_from_task_model
export GEMINI_API_KEY="your_gemini_api_key_here"
export GPT_API_KEY="your_gpt_api_key_here"
export LLAMA_TOKEN="your_hf_token_here"
bash ./run_generate_rationale.sh ${DEVICE} ${TASK}
cd path/to/larev-research/LAREV
bash ./run_generate_data_for_ranking_metric.sh ${TASK} 1
bash ./evaluate_ranking_data.sh ${DEVICE} ${BACKBONE_MODEL} ${TASK} 1 ${IRM_PENALTY} 1 ${LEAK_PROBE_PENALTY} 1
```

### Optional: Score rationale variants from REV metric

[REV](https://github.com/HanjieChen/REV) is used as a rationale evaluation metric for comparison of our work (LAREV). The command below reproduces the REV-based evaluation results reported in the paper.
```bash
cd path/to/larev-research/LAREV
bash ./evaluate_ranking_data.sh ${DEVICE} ${BACKBONE_MODEL} ${TASK} 0 0 0 0 0
```

---

## Acknowledgments

### Methodological Foundations

- This work builds upon the REV framework proposed by Chen et al., available at [HanjieChen/REV](https://github.com/HanjieChen/REV).
- [Integrated Gradients (IG)](https://arxiv.org/abs/1703.01365) (Sundararajan et al., 2017) is used as an attribution method to identify potential leakage terms.
- The construction of multiple environments in LAREV is inspired by [Invariant Risk Minimization (IRM)](https://arxiv.org/abs/1907.02893) (Arjovsky et al., 2019).

### Models and Resources

- The pretrained T5 infilling model used to construct baseline rationales for the ECQA task is obtained from [jifan-chen/QA-Verification-Via-NLI](https://github.com/jifan-chen/QA-Verification-Via-NLI)
- The pretrained T5 paraphraser used to construct baseline rationales for the e-SNLI task is obtained from [humarin/chatgpt_paraphraser_on_T5_base](https://huggingface.co/humarin/chatgpt_paraphraser_on_T5_base)

---

## Reproducibility Data and Models

All intermediate data and trained models required to reproduce the experimental results (T5-large on ECQA) in this thesis are archived in a single [file](https://drive.google.com/file/d/1DCp8QX1hq_2iOZULVy_akG5QQkYVuJsz/view?usp=sharing):

- `reproducibility_data.zip`

The archive preserves the original directory structure of the repository.
When extracted at the repository root, all data and model (T5-large on ECQA) files will be restored
to their expected locations.

> **Note**: The archive is stored in a private cloud location and is intended
> for long-term reproducibility and personal reference. Access can be granted
> upon request if needed.

To restore the data and models, extract the archive at the repository root:
```bash
cd path/to/larev-research/
unzip reproducibility_data.zip
```

---