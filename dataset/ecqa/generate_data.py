# Adapted from https://github.com/dair-iitd/ECQA-Dataset/tree/main?tab=readme-ov-file

import json
import pandas as pd
import os
import argparse


def normalize_sentence(sentences):
	s = sentences.strip()
	if len(s) == 0:
		return s
	if s[-1] not in ".?!":
		s = s + "."
	return s

def main():

	parser = argparse.ArgumentParser()

	parser.add_argument("--out_dir", required=True, type=str, help="Output directory for generated files.")
	args = parser.parse_args()

	current_path = os.path.dirname(os.path.abspath(__file__))

	decqa_file = os.path.join(current_path, 'ECQA-Dataset', 'ecqa.jsonl')
	cqa_file = os.path.join(current_path, '../', 'cqa', 'output', 'cqa.jsonl')

	label2idx = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4}


	lines = []
	with open(cqa_file) as f:
		for line in f:
			lines.append(json.loads(line))

	data = {}
	for line in lines:
		id = line['id']
		answerKey = line['answerKey']
		tmp_ques_obj = line['question']
		question = tmp_ques_obj['stem']
		concept = tmp_ques_obj['question_concept']
		choices_obj = tmp_ques_obj['choices']
		choices = ['' for i in range(5)]
		for choice in choices_obj:
			choices[label2idx[choice['label']]] = choice['text']
		answer = choices[label2idx[answerKey]]
		data[id] = {'question': question, 'answer': answer, 'choices': choices, 'concept': concept}

	print(len(data.keys()))

	lines = []
	with open(decqa_file) as f:
		for line in f:
			lines.append(json.loads(line))

	print(len(lines))

	twice = 0
	not_found = 0

	for line in lines:
		id = line['id']
		if ('positives' not in line) or ('negatives' not in line) or ('explanation' not in line):
			print(id, 'not in decqa')
			not_found += 1
		elif ('positives' in data[id]) and ('negatives' in data[id]) and ('freeflow' in data[id]):
			if line['explanation']!=data[id]['freeflow']:
				print(id, 'twice in decqa')
				twice += 1
				# print(line['positives'])
				# print(line['negatives'])
				# print(line['explanation'])
				# print(data[id]['positives'])
				# print(data[id]['negatives'])
				# print(data[id]['freeflow'])
		else:
			positives = line['positives']
			negatives = line['negatives']
			freeflow = line['explanation']
			data[id]['positives'] = positives
			data[id]['negatives'] = negatives
			data[id]['freeflow'] = freeflow

	print('# Not Found', not_found)
	print('# Twice', twice)

	total = 0
	q_ids = []
	q_concept = []
	q_text = []
	q_op1 = []
	q_op2 = []
	q_op3 = []
	q_op4 = []
	q_op5 = []
	q_ans = []
	q_ans = []
	taskA_pos = []
	taskA_neg = []
	taskB = []

	for id in data.keys():
		if ('positives' in data[id]) and ('negatives' in data[id]) and ('freeflow' in data[id]):
			total += 1
			q_ids.append(id)
			q_concept.append(data[id]['concept'])
			q_text.append(data[id]['question'])
			q_ans.append(data[id]['answer'])
			q_op1.append(data[id]['choices'][0])
			q_op2.append(data[id]['choices'][1])
			q_op3.append(data[id]['choices'][2])
			q_op4.append(data[id]['choices'][3])
			q_op5.append(data[id]['choices'][4])
			taskA_pos.append('\n'.join([normalize_sentence(p) for p in data[id]['positives']]))
			taskA_neg.append('\n'.join([normalize_sentence(n) for n in data[id]['negatives']]))
			taskB.append(data[id]['freeflow'])


	print(total)
	print(len(q_ids), len(q_concept), len(q_text), len(q_op1), len(q_op2), len(q_op3), len(q_op4), len(q_op5), len(q_ans), len(taskA_pos), len(taskA_neg), len(taskB))
	tmp_data = {'q_no':  q_ids,
			'q_concept': q_concept,
			'q_text': q_text,
			'q_op1': q_op1,
			'q_op2': q_op2,
			'q_op3': q_op3,
			'q_op4': q_op4,
			'q_op5': q_op5,
			'q_ans': q_ans,
			'taskA_pos': taskA_pos,
			'taskA_neg': taskA_neg,
			'taskB': taskB,
			}
	df = pd.DataFrame(tmp_data, columns = ['q_no', 'q_concept', 'q_text', 'q_op1', 'q_op2', 'q_op3', 'q_op4', 'q_op5',
										'q_ans', 'taskA_pos', 'taskA_neg', 'taskB'])
	# print(df.shape[0])
	# df = df[df['taskB'].str.split().str.len().gt(1)]
	# df = df[df['taskA_pos'].str.split().str.len().gt(1)]
	# df = df[df['taskA_neg'].str.split().str.len().gt(1)]
	# print(df.shape[0])

	os.makedirs(args.out_dir, exist_ok=True)
	df.to_csv(os.path.join(args.out_dir, 'ecqa.csv'), encoding='utf-8')

	split_path = os.path.join(current_path, 'ECQA-Dataset', 'author_split')
	SPLIT = ['train', 'val', 'test']
	for split_files in SPLIT:
		split_file = os.path.join(split_path, f"{split_files}_ids.txt")
		output_file = os.path.join(args.out_dir, f'ecqa_{split_files}.csv')

		ids = []
		with open(split_file) as f:
			for line in f:
				ids.append(line.strip())
		total = 0
		q_ids = []
		q_concept = []
		q_text = []
		q_op1 = []
		q_op2 = []
		q_op3 = []
		q_op4 = []
		q_op5 = []
		q_ans = []
		q_ans = []
		taskA_pos = []
		taskA_neg = []
		taskB = []

		for id in ids:
			if ('positives' in data[id]) and ('negatives' in data[id]) and ('freeflow' in data[id]):
				total += 1
				q_ids.append(id)
				q_concept.append(data[id]['concept'])
				q_text.append(data[id]['question'])
				q_ans.append(data[id]['answer'])
				q_op1.append(data[id]['choices'][0])
				q_op2.append(data[id]['choices'][1])
				q_op3.append(data[id]['choices'][2])
				q_op4.append(data[id]['choices'][3])
				q_op5.append(data[id]['choices'][4])
				taskA_pos.append('\n'.join([normalize_sentence(p) for p in data[id]['positives']]))
				taskA_neg.append('\n'.join([normalize_sentence(n) for n in data[id]['negatives']]))
				taskB.append(data[id]['freeflow'])


		print(total)
		print(len(q_ids), len(q_concept), len(q_text), len(q_op1), len(q_op2), len(q_op3), len(q_op4), len(q_op5), len(q_ans), len(taskA_pos), len(taskA_neg), len(taskB))
		tmp_data = {'q_no':  q_ids,
				'q_concept': q_concept,
				'q_text': q_text,
				'q_op1': q_op1,
				'q_op2': q_op2,
				'q_op3': q_op3,
				'q_op4': q_op4,
				'q_op5': q_op5,
				'q_ans': q_ans,
				'taskA_pos': taskA_pos,
				'taskA_neg': taskA_neg,
				'taskB': taskB,
				}
		df = pd.DataFrame(tmp_data, columns = ['q_no', 'q_concept', 'q_text', 'q_op1', 'q_op2', 'q_op3', 'q_op4', 'q_op5',
											'q_ans', 'taskA_pos', 'taskA_neg', 'taskB'])
		df.to_csv(output_file, encoding='utf-8')

if __name__ == "__main__":
    main()