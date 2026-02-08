# Adapted from:
# https://github.com/HanjieChen/REV/blob/main/src/rev/utils.py
import pandas as pd
import json
import random
import os
import logging

from transformers import AutoModelWithLMHead, AutoTokenizer

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def init_model(model_name: str, device, do_lower_case: bool = False):
    """
    Initialize a pre-trained LM
    :param model_name: from MODEL_CLASSES
    :param device: CUDA / CPU device
    :param do_lower_case: whether the model is lower cased or not
    :return: the model and tokenizer
    """

    if model_name == 'bart-large':
        model_name = 'facebook/'+model_name
    tokenizer = AutoTokenizer.from_pretrained(model_name, do_lower_case=do_lower_case) #, use_fast=False)
    model = AutoModelWithLMHead.from_pretrained(model_name)
    # import pdb; pdb.set_trace()
    model.to(device)
    model.eval()
    return tokenizer, model

def count_jsonl_lines(file_path):
    count = 0
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():  # Skip empty lines
                count += 1
    return count

def load_masked_rationales(ig_file, data_length, use_leak_probe, split_type):
    """Load masked rationales or return N/A placeholders."""
    # Only load masked rationales if using leak probe AND training split
    if not (use_leak_probe == 1 and split_type == 'train'):
        return ["N/A"] * data_length
    
    if not os.path.exists(ig_file):
        return ["N/A"] * data_length
    
    masked_samples = []
    with open(ig_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                result = json.loads(line)
                masked_samples.append(result['question_statement_text_masked'])
    
    return masked_samples


def load_data_ecqa(args, in_file, data_type=None, shuffle=True):
      current_path = os.path.dirname(os.path.abspath(__file__))
      file_name = in_file.split('/')[-1]
      if 'train' in file_name:
            split_type = 'train'
      elif 'val' in file_name:
            split_type = 'val'
      elif 'test' in file_name:
            split_type = 'test'
      if args.do_train or args.do_eval:
            ig_file = os.path.join(current_path, '../', 'generate_baseline_rationales', 'output', args.task, 'baseline_rationales_'+split_type+'_output_ig_'+args.model_name_or_path+'.jsonl')
            ig_file = os.path.normpath(ig_file)

            examples = []
            if data_type == 'regular' or data_type == 'filtered':
                  # read baseline rationales (b)
                  template_file = os.path.join(current_path, '../', 'generate_baseline_rationales', 'output', args.task, split_type+'_baseline_rationales_output.jsonl')
                  template_file = os.path.normpath(template_file)
                  samples = []
                  with open(template_file, 'r', encoding='utf-8') as json_file:
                        for json_str in json_file:
                              if not json_str.strip():
                                    continue
                              result = json.loads(json_str)
                              label = result['answer_text']
                              rat = result['question_statement_text']
                              samples.append((rat, label))

                  # read gold rationales (r)
                  df = pd.read_csv(in_file)
                  masked_samples  = load_masked_rationales(ig_file, len(df), args.use_leak_probe, split_type)

                  for i, row in df.iterrows():
                        pos_rat = row['taskA_pos'].replace('\n', " ")
                        answer = row['q_ans']
                        baseline_rat, label = samples[i]   # B, y
                        masked_rat = masked_samples[i]  # B_tilde
                        assert answer == label

                        # input to Φ: [rationale] R B [answer]
                        source_text = f"[rationale] {pos_rat} {baseline_rat} [answer]"
                        target_text = f"[answer] {answer} <eos>"

                        # input to ψ: [rationale] B_tilde [answer]
                        leak_source_text = f"[rationale] {masked_rat} [answer]"

                        examples.append((source_text, target_text, leak_source_text))

            elif data_type == 'temp':
                  file_length = count_jsonl_lines(in_file)
                  masked_samples = load_masked_rationales(ig_file, file_length, args.use_leak_probe, split_type)

                  with open(in_file, 'r', encoding='utf-8') as json_file:
                        for i, json_str in enumerate(json_file):
                              if not json_str.strip():
                                    continue
                              result = json.loads(json_str)
                              rat = result['question_statement_text']
                              answer = result['answer_text']
                              masked_rat = masked_samples[i]  # B_tilde
                              
                              # input to Φ: [rationale] B [answer]
                              source_text = f"[rationale] {rat} [answer]"
                              target_text = f"[answer] {answer} <eos>"

                              # input to ψ: [rationale] B_tilde [answer]
                              leak_source_text = f"[rationale] {masked_rat} [answer]"

                              examples.append((source_text, target_text, leak_source_text))
      if shuffle:
            random.shuffle(examples)
      return examples

def load_data_ecqa_irm(args, in_file, data_type=None, shuffle=True):
      if args.data_type != 'regular':
            raise ValueError("IRM training is only supported for regular model Φ. Please read the definition of Φ and Ψ in the paper.")
      current_path = os.path.dirname(os.path.abspath(__file__))
      file_name = in_file.split('/')[-1]
      
      # Determine split type from filename
      if 'train' in file_name:
            split_type = 'train'
      elif 'val' in file_name:
            split_type = 'val'
      elif 'test' in file_name:
            split_type = 'test'
      
      examples = []
      
      if args.do_train:
            # Path to multi_environments file
            multi_env_file = os.path.join(current_path, '../', 'generate_multi_environments', 'output', 'multi_environments_'+args.task.replace("_IRM", "")+'_'+args.model_name_or_path+'_'+split_type+'.jsonl') #can't use arg.task here because arg.task=='ECQA_IRM' now
            multi_env_file = os.path.normpath(multi_env_file)
            
            # Load multi_environments
            multi_envs = []
            if os.path.exists(multi_env_file):
                  with open(multi_env_file, 'r', encoding='utf-8') as f:
                        for line in f:
                              if line.strip():
                                    multi_envs.append(json.loads(line.strip()))
                  logger.info(f"Loaded {len(multi_envs)} multi_environments from {multi_env_file}")
            else:
                  raise FileNotFoundError(f"Multi environment file not found: {multi_env_file}")
            
            if data_type == 'regular':

                  # Concatenate gold rationale with multi_environments [r, E1], [r, E2], [r, E3]
                  if not os.path.exists(in_file):
                        logger.warning(f"Gold rationale file not found: {in_file}")
                        return examples
                  
                  df = pd.read_csv(in_file)
                  
                  # Check length mismatch and handle it
                  if len(df) != len(multi_envs):
                        raise ValueError(f"Length mismatch: {len(df)} rows in gold rationale file vs {len(multi_envs)} multi environments")
                  
                  for i, (_, row) in enumerate(df.iterrows()):
                        # Safety check: ensure multi_envs has enough elements
                        if i >= len(multi_envs):
                              logger.warning(f"Multi environments exhausted at index {i}. Stopping processing.")
                              raise IndexError(f"Length of multi environments exhausted at index {i}")

                        pos_rat = row['taskA_pos'].replace('\n', " ")
                        answer = row['q_ans']
                        q_id = i+1
                        env_data = multi_envs[i]
                        
                        # Verify answer consistency
                        if answer != env_data.get('answer_text', ''):
                              logger.warning(f"Answer mismatch at index {i}: {answer} vs {env_data.get('answer_text', '')}")
                              raise ValueError(f"Answer mismatch at index {i}")
                        
                        # [r, E1]: Gold + Original baseline
                        if 'baseline' in env_data:
                              baseline_rat = env_data['baseline']
                              masked_rat = env_data['masked']
                              env_id = 1
                              # input to Φ: [rationale] R B [answer]
                              source_text = f"[rationale] {pos_rat} {baseline_rat} [answer]"
                              target_text = f"[answer] {answer} <eos>"
                              # input to ψ: [rationale] B_tilde [answer]
                              leak_source_text = f"[rationale] {masked_rat} [answer]"

                              examples.append((source_text, target_text, q_id, env_id, leak_source_text))
                        
                        # [r, E2]: Gold + Masked baseline
                        if 'masked' in env_data:
                              masked_rat = env_data['masked']
                              env_id = 2
                              # input to Φ: [rationale] R B [answer]
                              source_text = f"[rationale] {pos_rat} {masked_rat} [answer]"
                              target_text = f"[answer] {answer} <eos>"
                              # input to ψ: [rationale] B_tilde [answer]
                              leak_source_text = f"[rationale] {masked_rat} [answer]"

                              examples.append((source_text, target_text, q_id, env_id, leak_source_text))

                        # [r, E3]: Gold + Antonym baseline
                        if 'antonym' in env_data:
                              masked_rat = env_data['masked']
                              env_id = 3
                              antonym_rat = env_data['antonym']
                              # input to Φ: [rationale] R B [answer]
                              source_text = f"[rationale] {pos_rat} {antonym_rat} [answer]"
                              target_text = f"[answer] {answer} <eos>"
                              # input to ψ: [rationale] B_tilde [answer]
                              leak_source_text = f"[rationale] {masked_rat} [answer]"

                              examples.append((source_text, target_text, q_id, env_id, leak_source_text))
            elif data_type == 'temp':

                  raise NotImplementedError("Temporary model training is not supported for ECQA_IRM currently.")

      if shuffle:
            random.shuffle(examples)
      
      logger.info(f"Loaded {len(examples)} IRM examples (data_type={data_type})")
      return examples

def load_data_esnli(args, in_file, data_type=None, shuffle=True):
      current_path = os.path.dirname(os.path.abspath(__file__))
      file_name = in_file.split('/')[-1]
      if 'train' in file_name:
            split_type = 'train'
      elif 'val' in file_name:
            split_type = 'val'
      elif 'test' in file_name:
            split_type = 'test'
      if args.do_train or args.do_eval:
            ig_file = os.path.join(current_path, '../', 'generate_baseline_rationales', 'output', args.task, 'baseline_rationales_'+split_type+'_output_ig_'+args.model_name_or_path+'.jsonl')
            ig_file = os.path.normpath(ig_file)

            examples = []
            if data_type == 'regular':
                  # read baseline rationales (b)
                  template_file = os.path.join(current_path, '../', 'generate_baseline_rationales', 'output', args.task, 'baseline_rationales_'+split_type+'_output.jsonl')
                  template_file = os.path.normpath(template_file)
                  samples = []
                  with open(template_file, 'r', encoding='utf-8') as json_file:
                        for json_str in json_file:
                              if not json_str.strip():
                                    continue
                              result = json.loads(json_str)
                              label = result['answer_text']
                              rat = result['question_statement_text']
                              samples.append((rat, label))

                  file_length = count_jsonl_lines(in_file)
                  masked_samples = load_masked_rationales(ig_file, file_length, args.use_leak_probe, split_type)

                  with open(in_file, 'r', encoding='utf-8') as json_file:
                        for i, json_str in enumerate(json_file):
                              if not json_str.strip():
                                    continue
                              result = json.loads(json_str)
                              pos_rat = result['rationale']
                              answer = result['answer_text']
                              baseline_rat, label = samples[i]   # B, y
                              masked_rat = masked_samples[i]  # B_tilde
                              assert answer == label
                              
                              # input to Φ: [rationale] R B [answer]
                              source_text = f"[rationale] {pos_rat} {baseline_rat} [answer]"
                              target_text = f"[answer] {answer} <eos>"

                              # input to ψ: [rationale] B_tilde [answer]
                              leak_source_text = f"[rationale] {masked_rat} [answer]"

                              examples.append((source_text, target_text, leak_source_text))

            elif data_type == 'temp':
                  file_length = count_jsonl_lines(in_file)
                  masked_samples = load_masked_rationales(ig_file, file_length, args.use_leak_probe, split_type)

                  with open(in_file, 'r', encoding='utf-8') as json_file:
                        for i, json_str in enumerate(json_file):
                              if not json_str.strip():
                                    continue
                              result = json.loads(json_str)
                              rat = result['question_statement_text']
                              answer = result['answer_text']
                              masked_rat = masked_samples[i]  # B_tilde
                              
                              # input to Φ: [rationale] B [answer]
                              source_text = f"[rationale] {rat} [answer]"
                              target_text = f"[answer] {answer} <eos>"

                              # input to ψ: [rationale] B_tilde [answer]
                              leak_source_text = f"[rationale] {masked_rat} [answer]"

                              examples.append((source_text, target_text, leak_source_text))

      if shuffle:
            random.shuffle(examples)
      return examples

def load_data_esnli_irm(args, in_file, data_type=None, shuffle=True):
      if args.data_type != 'regular':
            raise ValueError("IRM training is only supported for regular model Φ. Please read the definition of Φ and Ψ in the paper.")
      current_path = os.path.dirname(os.path.abspath(__file__))
      file_name = in_file.split('/')[-1]
      
      # Determine split type from filename
      if 'train' in file_name:
            split_type = 'train'
      elif 'val' in file_name:
            split_type = 'val'
      elif 'test' in file_name:
            split_type = 'test'
      
      examples = []
      
      if args.do_train:
            # Path to multi_environments file
            multi_env_file = os.path.join(current_path, '../', 'generate_multi_environments', 'output', 'multi_environments_'+args.task.replace("_IRM", "")+'_'+args.model_name_or_path+'_'+split_type+'.jsonl') #can't use arg.task here because arg.task=='ESNLI_IRM' now
            multi_env_file = os.path.normpath(multi_env_file)
            
            # Load multi_environments
            multi_envs = []
            if os.path.exists(multi_env_file):
                  with open(multi_env_file, 'r', encoding='utf-8') as f:
                        for line in f:
                              if line.strip():
                                    multi_envs.append(json.loads(line.strip()))
                  logger.info(f"Loaded {len(multi_envs)} multi_environments from {multi_env_file}")
            else:
                  logger.warning(f"Multi environment file not found: {multi_env_file}")
                  return examples
            
            if data_type == 'regular':

                  # Concatenate gold rationale with multi_environments [r, E1], [r, E2], [r, E3]
                  if not os.path.exists(in_file):
                        logger.warning(f"Gold rationale file not found: {in_file}")
                        return examples
                  
                  samples = []
                  with open(in_file, 'r', encoding='utf-8') as json_file:
                        for i, json_str in enumerate(json_file):
                              if not json_str.strip():
                                    continue
                              if i >= len(multi_envs):
                                    logger.warning(f"Multi environments exhausted at index {i}. Stopping processing.")
                                    raise IndexError(f"Length of multi environments exhausted at index {i}")
                              result = json.loads(json_str)
                              pos_rat = result['rationale']
                              answer = result['answer_text']
                              q_id = i+1
                              env_data = multi_envs[i]
                        
                              # Verify answer consistency
                              if answer != env_data.get('answer_text', ''):
                                    logger.warning(f"Answer mismatch at index {i}: {answer} vs {env_data.get('answer_text', '')}")
                                    raise ValueError(f"Answer mismatch at index {i}")
                              
                              # [r, E1]: Gold + Original baseline
                              if 'baseline' in env_data:
                                    baseline_rat = env_data['baseline']
                                    masked_rat = env_data['masked']
                                    env_id = 1
                                    # input to Φ: [rationale] R B [answer]
                                    source_text = f"[rationale] {pos_rat} {baseline_rat} [answer]"
                                    target_text = f"[answer] {answer} <eos>"
                                    # input to ψ: [rationale] B_tilde [answer]
                                    leak_source_text = f"[rationale] {masked_rat} [answer]"

                                    examples.append((source_text, target_text, q_id, env_id, leak_source_text))
                              
                              # [r, E2]: Gold + Masked baseline
                              if 'masked' in env_data:
                                    masked_rat = env_data['masked']
                                    env_id = 2
                                    # input to Φ: [rationale] R B [answer]
                                    source_text = f"[rationale] {pos_rat} {masked_rat} [answer]"
                                    target_text = f"[answer] {answer} <eos>"
                                    # input to ψ: [rationale] B_tilde [answer]
                                    leak_source_text = f"[rationale] {masked_rat} [answer]"

                                    examples.append((source_text, target_text, q_id, env_id, leak_source_text))

                              # [r, E3]: Gold + Antonym baseline
                              if 'antonym' in env_data:
                                    masked_rat = env_data['masked']
                                    env_id = 3
                                    antonym_rat = env_data['antonym']
                                    # input to Φ: [rationale] R B [answer]
                                    source_text = f"[rationale] {pos_rat} {antonym_rat} [answer]"
                                    target_text = f"[answer] {answer} <eos>"
                                    # input to ψ: [rationale] B_tilde [answer]
                                    leak_source_text = f"[rationale] {masked_rat} [answer]"

                                    examples.append((source_text, target_text, q_id, env_id, leak_source_text))
            elif data_type == 'temp':

                  raise NotImplementedError("Temporary model training is not supported for ESNLI_IRM currently.")
      if shuffle:
            random.shuffle(examples)
      
      logger.info(f"Loaded {len(examples)} IRM examples (data_type={data_type})")
      return examples

def load_data_leak_probe(in_file):
      """
      Load leakage probing data (masked baseline rationales) for training a leakage probing model ψ.
      """
      logger.info("Loading leakage probing data from: " + in_file)

      # read baseline rationales (b)
      samples = []
      with open(in_file, 'r') as json_file:
            json_list = list(json_file)
            for json_str in json_list:
                  result = json.loads(json_str)
                  label = result['answer_text']
                  masked_rat = result['question_statement_text_masked']
                  tilde_B = masked_rat # tilde_B is the already masked baseline rationale
                  samples.append((f"[rationale] {tilde_B} [answer]", f"[answer] {label} <eos>"))

      return samples

def load_data_ranking(args, in_file, data_type=None, shuffle=False):
      examples = []
      if not os.path.exists(in_file):
            logger.error(f"Ranking file not found: {in_file}")
            return examples

      if getattr(args, "ranking_type", None) not in ["gold", "gold_leaky", "vacuous", "leaky", "truncated_gold_80", "truncated_gold_50", "gold_noise", "shuffled_gold"]:
            logger.error(f"Invalid ranking type: {getattr(args, 'ranking_type', None)}")
            return examples

      valid_types = {"gold", "leaky", "gold_leaky", "vacuous", "truncated_gold_80", "truncated_gold_50", "gold_noise", "shuffled_gold"}
      wanted_type = args.ranking_type

      with open(in_file, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                  line = line.strip()
                  if not line:
                        continue
                  try:
                        obj = json.loads(line)
                  except json.JSONDecodeError as e:
                        logger.warning(f"[ranking] JSON decode error at line {line_num}: {e}")
                        continue

                  # minimal field checks
                  if "type" not in obj or obj["type"] not in valid_types:
                        logger.warning(f"[ranking] Missing/invalid 'type' at line {line_num}")
                        continue
                  if obj["type"] != wanted_type:
                        continue
                  if "answer_text" not in obj:
                        logger.warning(f"[ranking] Missing 'answer_text' at line {line_num}")
                        continue

                  answer = obj["answer_text"]

                  if data_type == "regular":
                        # need both R and B
                        R = obj.get("rationale", "")
                        B = obj.get("baseline_rationale", "")
                        input_text = f"[rationale] {R} {B} [answer]"
                        label_text = f"[answer] {answer} <eos>"
                        examples.append((input_text, label_text))

                  elif data_type == "temp":
                        # only B is present in temp file under the key 'baseline_rationale'
                        B = obj.get("baseline_rationale", "")
                        input_text = f"[rationale] {B} [answer]"
                        label_text = f"[answer] {answer} <eos>"
                        examples.append((input_text, label_text))

                  else:
                        logger.warning(f"[ranking] Unsupported data_type '{data_type}' at line {line_num}; skip.")

      if shuffle:
            random.shuffle(examples)

      logger.info(f"Loaded {len(examples)} ranking examples from {in_file} (data_type={data_type}, ranking_type={args.ranking_type})")
      return examples