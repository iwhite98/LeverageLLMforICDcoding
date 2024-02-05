from transformers import T5Tokenizer, T5ForConditionalGeneration, DataCollatorWithPadding, Trainer, TrainingArguments, DataCollatorForSeq2Seq
import tqdm
import fire
from datasets import load_dataset, Dataset
import torch
import pandas as pd
import os
import torch.distributed as dist
import pickle
import json
from compute_metric import calculate_eval_metrics
import numpy as np
import re

#dist.init_process_group('gloo', rank = 3, world_size = 4)
from nltk.tokenize import sent_tokenize, RegexpTokenizer

text_tokenizer = RegexpTokenizer(r'\w+')
RECORD_SEPARATOR = "\n\n"

CHAPTER = 1
THREE_CHARACTER = 2
FULL = 3
n_not_found = 0


def clean_data(text):
    text = text.split('[**')
    post = []
    for t in text:
        if '**]' in t:
            t = t.split('**]')
            post.append(t[1])
        else:
            post.append(t)
    post = ''.join(post)

    post = post.replace('\n\n', ' ')
    post = post.replace('\n', ' ')

    ppost = []
    for i in range(len(post)):
        ppost.append(post[i])
        if (i+1) < len(post) and post[i].isalpha() and post[i+1].isdigit():
            ppost.append(' ')
        elif (i+1) < len(post) and post[i].isdigit() and post[i+1].isalpha():
            ppost.append(' ')

    ppost = ''.join(ppost)
    ppost = ''.join([e if e.isalnum() or e == '.' else ' ' for e in ppost])

    while '  ' in ppost:
        ppost = ppost.replace('  ', ' ')

    ppost = ppost.split(' ')
    output = []
    length = 0

    for word in ppost:
        if word.isalpha():
            tokens = [token.lower() for token in text_tokenizer.tokenize(word)]
            length += len(tokens)
            sent = " ".join(tokens)
            output.append(sent)
        else:
            length += 1
            output.append(word)
    output = " ".join(output)

    return output

def text_sub(text):
	output= re.sub('[=+,#/\?:^.@*\"※~ㆍ!』‘|\(\)\[\]`\'…》\”\“\’·]', ' ', text)
	output = re.sub('\s+',' ', output)
	output = re.sub('--|__|==', '', output)
	output = re.sub('[\(;*?\)\[\]]', '', output)
	output = re.sub('\t', ' ', output)
	return output

def run(
	data_path: str = '',
	model_dir: str = '',
	output_dir: str='',
	train_mode: str ='',
	loss_mode: str='',
	model_name: str=''
	):
	if torch.cuda.is_available():
		device = 'cuda'
	else:
		device = 'cpu'
	
	if model_name == 't5-base':	
		if torch.cuda.device_count() == 2:
			device_map = {0: [0, 1, 2,3,4,5],
			1: [6,7,8,9,10,11]}
		elif torch.cuda.device_count() == 4:
			device_map = {0: [0, 1, 2],
			1: [3,4,5],
			2: [6,7,8],
			3: [9,10,11]}
	elif model_name == 't5-large' or model_name == 't5-3b':	
		if torch.cuda.device_count() == 2:
			device_map = {0: [0, 1, 2,3,4,5],
			1: [6,7,8,9,10,11]}
		elif torch.cuda.device_count() == 4:
			device_map = {0: [0, 1, 2,3,4,5],
			1: [6,7,8,9,10,11],
			2: [12,13,14,15,16,17],
			3: [18,19,20,21,22,23]}

	tokenizer = T5Tokenizer.from_pretrained(model_name, device_map=device_map)
	with open(output_dir + '/vocab.pkl', 'rb') as f:
		vocab = pickle.load(f)
	if train_mode == 'decode':
		if loss_mode == 'both':
			from trainer.trainer_att_dec import MultiLabelTrainer
			from net.net_decode import Net
			model = Net(model_name, tokenizer)
		elif loss_mode == 'att':
			from trainer.trainer_att import MultiLabelTrainer
			from net.net_decode import Net
			model = Net(model_name, tokenizer)
		elif loss_mode == 'label':
			from trainer.trainer_label import MultiLabelTrainer
			from net.net_decode_label_loss import Net
			model = Net(model_name, tokenizer, vocab)
	elif train_mode == 'encode_decode':
		from net.net_encode_decode import Net
		model = Net(model_name)
	elif train_mode == 'encode':
		from net.net_encode_label_loss import Net
		model = Net(model_name, tokenizer, vocab)
		if loss_mode == 'label':
			from trainer.trainer_label import MultiLabelTrainer
	elif train_mode == 'encode_parallel_att':
		if loss_mode == 'label':
			from trainer.trainer_label_parallel import MultiLabelTrainer
			#from net.net_encode_parallel_att_label_loss import Net
			from net.net_encode_parallel_vocab import Net
			model = Net(model_name, tokenizer, vocab)
	elif train_mode == 'encode_parallel_att_lstm':
		if loss_mode == 'label':
			from trainer.trainer_label_parallel import MultiLabelTrainer
			#from trainer.trainer_cr_parallel import MultiLabelTrainer
			from net.net_encode_parallel_att_label_loss_lstm import Net
			model = Net(model_name, tokenizer, vocab)
	elif train_mode == 'decode_parallel_att':
		from net.net_decode_parallel_att import Net
		model = Net(model_name, tokenizer)
	elif train_mode == 'decode_parallel':
		from net.net_decode_parallel import Net
		model = Net(model_name)
	elif train_mode == 'decode_att':
		from net.net_decode_parallel_att import Net  ## argument None
		model = Net(model_name, tokenizer)
	elif train_mode == 'org':
		#from net.net_org_parallel_att import Net  ## argument None
		from net.net_org import Net
		#from net.net_org_vocab import Net
		model = Net(model_name, tokenizer, vocab)
		from trainer.trainer_org import MultiLabelTrainer
	elif train_mode == 'org_att':
		#from net.net_org_parallel_att import Net  ## argument None
		if loss_mode == 'label':
			from net.net_org_att import Net
			model = Net(model_name, tokenizer, vocab)
			from trainer.trainer_label import MultiLabelTrainer
		elif loss_mode == 'label_wot_desc':
			from net.net_org_att import Net
			model = Net(model_name, tokenizer, vocab)
			from trainer.trainer_wot_desc_loss import MultiLabelTrainer
	elif train_mode == 'org_parallel_vocab':
		#from net.net_org_parallel_att import Net  ## argument None
		from net.net_org_parallel_vocab import Net
		model = Net(model_name, tokenizer, vocab)
		from trainer.trainer_parallel import MultiLabelTrainer
	elif train_mode == 'org_parallel_desc':
		#from net.net_org_parallel_att import Net  ## argument None
		from net.net_org_parallel_desc import Net
		model = Net(model_name, tokenizer, vocab)
		from trainer.trainer_parallel import MultiLabelTrainer
	elif train_mode == 'org_att_parallel_vocab':
		#from net.net_org_parallel_att import Net  ## argument None
		from net.net_org_att_parallel_vocab_test import Net
		model = Net(model_name, tokenizer, vocab)
		from trainer.trainer_label_parallel import MultiLabelTrainer

	max_source_length = 2048
	max_target_length = 512

	instruction = f'The following note is the discharge summary of a patient. Provide all ICD-9 codes applicable to the patient.'
	
	def generate_prompt(data_point):
		temp1 = f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.
		### Instruction:
		{instruction}
		### Input:
		{data_point["input"]}
		### Response:
		"""
		temp2 = f"""{data_point["output"]}"""
		return temp1, temp2

	def generate_and_tokenize(data_point):
		#instruction = data_point['instruction']
		input = str(data_point['input'])
		output = data_point['output']

		########only for description reference dataset
		output_ = []
		for line in output.split('\n'):
			for word in line.split(' '):
				if word in vocab.label2index.keys():
					output_.append(word)
		output = ','.join(output_)
		data_point['output'] = output

		input, output = generate_prompt(data_point)

		clean_input = text_sub(clean_data(input))

		code_labels = [0] * 50
		code_list = []
		#for line in output.split('\n'):
		for line in output.split(','):
			code = line.split(' : ')[0]
			code_list.append(code)
			code_labels[vocab.label2index[code]] = 1.0
		clean_output = output

		source_encoding = tokenizer(
			clean_input,
			max_length=max_source_length,
			padding = 'max_length',
		)

	
		target_encoding = tokenizer(
			clean_output,
			max_length=max_target_length,
			padding = 'max_length',
		)
	
		data = dict()
		len_input = len(source_encoding['input_ids'])
		len_output = len(target_encoding['input_ids'])
		possible_length = max_source_length - len_output
	
		
		source_encoding['input_ids'] = source_encoding['input_ids']
		source_encoding['attention_mask'] = source_encoding['attention_mask']

		if len_input > possible_length:
			data['input_ids'] = source_encoding['input_ids'][:possible_length - 7] + source_encoding['input_ids'][len_input-7:len_input]
			data['attention_mask'] = source_encoding['attention_mask'][:possible_length - 7] + source_encoding['attention_mask'][len_input-7:len_input]
		else:
			data['input_ids'] = source_encoding['input_ids'][:len_input-7] + [0]*(possible_length - len_input) + source_encoding['input_ids'][len_input-7:len_input]
			data['attention_mask'] = source_encoding['attention_mask'][:len_input-7] + [0]*(possible_length - len_input) + source_encoding['attention_mask'][len_input-7:len_input]

		labels = torch.tensor(target_encoding['input_ids'])

		labels[labels == tokenizer.pad_token_id] = -100
		

		data['decoder_labels'] = labels.squeeze()[:max_target_length].to(device)
		data['training'] = False
		data["labels"] = torch.tensor(code_labels).to(device)
		data['input_ids'] = torch.tensor(data['input_ids'])
		data['attention_mask'] = torch.tensor(data['attention_mask'])
		return data


	
	data_files = {"test" : "test.csv"}
	full_data = load_dataset(data_path, data_files = data_files)
	test_data = full_data['test'].shuffle().map(generate_and_tokenize).select_columns(['input_ids', 'attention_mask', 'labels', 'decoder_labels', 'training'])
	
	if torch.cuda.device_count() > 1:
		# keeps Trainer from trying its own DataParallelism when more than 1 gpu is available
		model.is_parallelizable = True
		model.model_parallel = True
		#model.model_t5.parallelize(device_map)
	
	print('load model')
	model.load_state_dict(torch.load(model_dir + '/pytorch_model.bin'))

	print('---------Label prediction--------')
	true_label = []
	pred_label = []
	result = dict()
	with torch.no_grad():
		for idx, data in enumerate(test_data):

			input_ids = torch.tensor(data['input_ids']).unsqueeze(0).to('cuda')
			attention_mask = torch.tensor(data['attention_mask']).unsqueeze(0).to('cuda')
			decoder_labels = torch.tensor(data['decoder_labels']).unsqueeze(0).to('cuda')
			#decoder_input_ids = torch.tensor(data['decoder_input_ids']).unsqueeze(0).to('cuda')
			training = torch.tensor(data['training']).unsqueeze(0)
			labels = torch.tensor(data['labels']).unsqueeze(0).to('cuda')
			'''
			if train_mode == 'decode_parallel_att' or train_mode == 'decode_att' or train_mode == 'decode_parallel':
				output, output_dec = model(input_ids, attention_mask, decoder_input_ids, decoder_labels, labels, None, None)
			else:
				output, output_dec = model(input_ids, attention_mask, decoder_input_ids, decoder_labels, labels)
			'''
			if train_mode == 'org' :
				output_dec = model(input_ids, attention_mask, decoder_labels, labels, training)
			elif train_mode == 'decode_parallel_att' or train_mode == 'decode_att' or train_mode == 'decode_parallel':
				output, output_dec = model(input_ids, attention_mask, decoder_labels, labels, training, None, None)
				output = torch.sigmoid(output)
				true_label.append(labels.squeeze().detach().cpu().tolist())
				pred_label.append(output.squeeze().detach().cpu().tolist())
			elif train_mode == 'org_parallel_vocab' or train_mode == 'org_parallel_desc':
				output_dec = model(input_ids, attention_mask, decoder_labels, labels, training)
			else:
				output, output_dec = model(input_ids, attention_mask, decoder_labels, labels, training)
				output = torch.sigmoid(output)
				true_label.append(labels.squeeze().detach().cpu().tolist())
				pred_label.append(output.squeeze().detach().cpu().tolist())
			
			if loss_mode != 'label_wot_desc':
				output_dec = output_dec.argmax(-1)
				output_dec = tokenizer.batch_decode(output_dec)
				true_output = torch.tensor(data['decoder_labels'])
				true_idx = true_output != -100
				true_output = tokenizer.decode(true_output[true_idx])
				result_ = {
					'true_output' : true_output,
					'output' : output_dec[0]
				}
				result[idx] = result_
				#print(true_output)
				#print(output_dec)
			if loss_mode != 'label_wot_desc':
				with open(output_dir + '/result.json', 'w') as f:
					json.dump(result, f)
	if train_mode != 'org' and train_mode != 'org_parallel_vocab' and train_mode != 'org_parallel_desc':
		print(calculate_eval_metrics(true_label, pred_label, True))
		with open(output_dir + '/true_label.pkl', 'wb') as f:
			pickle.dump(true_label, f)
		
		with open(output_dir + '/pred_label.pkl', 'wb') as f:
			pickle.dump(pred_label, f)
	


if __name__ == "__main__":
	fire.Fire(run)
