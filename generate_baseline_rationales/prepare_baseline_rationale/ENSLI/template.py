# Adapted from:
# https://github.com/HanjieChen/REV/blob/main/src/esnli_baseline/template.py

import csv
import os
import sys
import json
import pandas as pd

##########################
# We only use 10000 samples for esnli train set
# We only use 2000 samples for esnli val set
# We only use 2000 samples for esnli test set
##########################


def normalize_rationale(rationale):
        """Normalize rationale to start with capital letter and end with period."""
        rationale = rationale.strip()
        # Remove newline characters
        rationale = rationale.replace('\n', ' ')
        if not rationale:
                return rationale
        # Capitalize first letter
        rationale = rationale[0].upper() + rationale[1:] if len(rationale) > 1 else rationale.upper()
        # Add period if missing
        if not rationale.endswith('.'):
                rationale += '.'
        return rationale


def main():
        csv.field_size_limit(sys.maxsize)
        current_path = os.path.dirname(os.path.abspath(__file__))

        task = 'esnli'
        split = ['train', 'val', 'test']
        for s in split:

                input_file = os.path.join(current_path, '../', '../', '../', 'dataset', task, task+'_'+s+'.csv')
                input_file = os.path.normpath(input_file)
                output_file = os.path.join(current_path, '../', '../', '../', 'dataset', task, 'output', task+'_'+s+'_templated.jsonl')
                output_file = os.path.normpath(output_file)
                if not os.path.exists(input_file):
                        raise FileNotFoundError(f"Input file not found: {input_file}")
                os.makedirs(os.path.dirname(output_file), exist_ok=True)
                
                df = pd.read_csv(input_file)
                examples = []
                for index, row in df.iterrows():
                        if s == 'train':
                                if index == 10000:
                                        break
                        elif s == 'val':
                                if index == 2000:
                                        break
                        elif s == 'test':
                                if index == 2000:
                                        break
                        label = str(row['gold_label'])
                        premise = normalize_rationale(str(row['Sentence1']))
                        hypothesis = normalize_rationale(str(row['Sentence2']))
                        rationale = normalize_rationale(str(row['Explanation_1']))
                        if label == 'entailment':
                                question_statement_text = premise + ' implies ' + hypothesis
                        elif label == 'contradiction':
                                question_statement_text = premise + ' contradicts ' + hypothesis
                        elif label == 'neutral':
                                question_statement_text = premise + ' is not related to ' + hypothesis
                        else:
                                raise ValueError(f"Unknown label: {label} on row {index}")
                        examples.append(
                                {
                                        'question_text': premise + ' ' + hypothesis,
                                        'answer_text': label,
                                        'question_statement_text': question_statement_text,
                                        'rationale': rationale
                                }
                        )

                with open(output_file, "w") as f_out:
                        for example in examples:
                                f_out.write(
                                        json.dumps(example)
                                        + "\n"
                                )

if __name__ == "__main__":
    main()
