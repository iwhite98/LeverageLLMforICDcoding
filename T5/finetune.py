from transformers import T5Tokenizer, T5ForConditionalGeneration, DataCollatorWithPadding, Trainer, TrainingArguments, DataCollatorForSeq2Seq
import fire
from datasets import load_dataset, Dataset
import torch
import pandas as pd
import os
import torch.distributed as dist
from vocab import Vocab
import pickle
#dist.init_process_group('gloo', rank = 3, world_size = 4)
from nltk.tokenize import sent_tokenize, RegexpTokenizer
import re
from compute_metric import *

from peft import (
	LoraConfig,
	get_peft_model,
	get_peft_model_state_dict,
	prepare_model_for_int8_training,
	PeftModel
)

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
	output_dir: str = '',
	num_epoch: int = 3,
	train_mode: str ='',
	loss_mode: str = '',
	model_name: str='',
	batch_size: int = 4
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
		elif torch.cuda.device_count() == 8:
			device_map = {0: [0, 1, 2],
			1: [3,4,5],
			2: [6,7,8],
			3: [9,10,11],
			4: [12,13,14],
			5: [15,16,17],
			6: [18,19,20],
			7: [21,22,23]}
	tokenizer = T5Tokenizer.from_pretrained(model_name, device_map=device_map)
	#vocab = Vocab('../dataset/code_description/50')
	with open(output_dir + '/vocab.pkl', 'rb') as f:
		vocab = pickle.load(f)
		#pickle.dump(vocab, f)
	from net.net import Net
	model = Net(model_name, tokenizer, vocab)
	if loss_mode == 'label':
		from trainer.trainer import MultiLabelTrainer
		
	
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
		temp1_ = f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.
		### Instruction:
		{instruction}
		### Input:
		{data_point["input_ref"]}
		### Response:
		"""
		temp2 = f"""{data_point["output"]}"""
		return temp1, temp1_, temp2

	def generate_and_tokenize(data_point):
		#instruction = data_point['instruction']
		input = str(data_point['input'])
		output = data_point['output']

		output_ = []
		for line in output.split('\n'):
			for word in line.split(' '):
				if word in vocab.label2index.keys():
					output_.append(word)
		output = ','.join(output_)
		data_point['output'] = output

		input, input_ref, output = generate_prompt(data_point)

		clean_input = text_sub(clean_data(input))
		clean_input_ref = text_sub(clean_data(input_ref))

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

		source_encoding_ref = tokenizer(
			clean_input_ref,
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
		len_input_ref = len(source_encoding_ref['input_ids'])
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

		if len_input_ref > possible_length:
			data['input_ids_ref'] =source_encoding_ref['input_ids'][:possible_length - 7] + source_encoding_ref['input_ids'][len_input-7:len_input]
			data['attention_mask_ref'] = source_encoding_ref['attention_mask'][:possible_length - 7] + source_encoding_ref['attention_mask'][len_input-7:len_input]
		else:
			data['input_ids_ref'] = source_encoding_ref['input_ids'][:len_input-7] + [0]*(possible_length - len_input) + source_encoding_ref['input_ids'][len_input-7:len_input]
			data['attention_mask_ref'] = source_encoding_ref['attention_mask'][:len_input-7] + [0]*(possible_length - len_input) + source_encoding_ref['attention_mask'][len_input-7:len_input]

		
		data['input_ids_ref'] = [0] * (2048 - len(data['input_ids_ref'])) + data['input_ids_ref']
		data['attention_mask_ref'] = [0] * (2048 - len(data['attention_mask_ref'])) + data['attention_mask_ref']

		labels = torch.tensor(target_encoding['input_ids'])

		labels[labels == tokenizer.pad_token_id] = -100
		

		data['decoder_labels'] = labels.squeeze()[:max_target_length].to(device)
		data['training'] = True
		data["labels"] = torch.tensor(code_labels).to(device)
		data['input_ids'] = torch.tensor(data['input_ids']).to(device)
		data['attention_mask'] = torch.tensor(data['attention_mask']).to(device)
		data['input_ids_ref'] = torch.tensor(data['input_ids_ref']).to(device)
		data['attention_mask_ref'] = torch.tensor(data['attention_mask_ref']).to(device)
		return data

	def generate_and_tokenize_test(data_point):
		#instruction = data_point['instruction']
		input = str(data_point['input'])
		output = data_point['output']

		output_ = []
		for line in output.split('\n'):
			for word in line.split(' '):
				if word in vocab.label2index.keys():
					output_.append(word)
		output = ','.join(output_)
		data_point['output'] = output

		input, input_ref, output = generate_prompt(data_point)

		clean_input = text_sub(clean_data(input))
		clean_input_ref = text_sub(clean_data(input_ref))

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

		source_encoding_ref = tokenizer(
			clean_input_ref,
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
		len_input_ref = len(source_encoding_ref['input_ids'])
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

		if len_input_ref > possible_length:
			data['input_ids_ref'] = source_encoding_ref['input_ids'][:possible_length - 7] + source_encoding_ref['input_ids'][len_input-7:len_input]
			data['attention_mask_ref'] =  source_encoding_ref['attention_mask'][:possible_length - 7] + source_encoding_ref['attention_mask'][len_input-7:len_input]
		else:
			data['input_ids_ref'] =  source_encoding_ref['input_ids'][:len_input-7] + [0]*(possible_length - len_input) + source_encoding_ref['input_ids'][len_input-7:len_input]
			data['attention_mask_ref'] =  source_encoding_ref['attention_mask'][:len_input-7] + [0]*(possible_length - len_input) + source_encoding_ref['attention_mask'][len_input-7:len_input]
		
		data['input_ids_ref'] = [0] * (2048 - len(data['input_ids_ref'])) + data['input_ids_ref']
		data['attention_mask_ref'] = [0] * (2048 - len(data['attention_mask_ref'])) + data['attention_mask_ref']

		labels = torch.tensor(target_encoding['input_ids'])

		labels[labels == tokenizer.pad_token_id] = -100
		

		data['decoder_labels'] = labels.squeeze()[:max_target_length].to(device)
		data['training'] = False
		data["labels"] = torch.tensor(code_labels).to(device)
		data['input_ids'] = torch.tensor(data['input_ids']).to(device)
		data['attention_mask'] = torch.tensor(data['attention_mask']).to(device)
		data['input_ids_ref'] = torch.tensor(data['input_ids_ref']).to(device)
		data['attention_mask_ref'] = torch.tensor(data['attention_mask_ref']).to(device)
		return data

	
	data_files = {"train" : "train.csv", "valid" : 'valid.csv'}
	full_data = load_dataset(data_path, data_files = data_files)
	train_data = full_data['train'].shuffle().map(generate_and_tokenize).select_columns(['input_ids', 'attention_mask', 'labels', 'decoder_labels', 'training'])
	val_data = full_data['valid'].shuffle().map(generate_and_tokenize_test).select_columns(['input_ids', 'attention_mask', 'labels', 'decoder_labels', 'training'])
	
	val_set_size = len(val_data)
	
	lr = 1e-4
	data_collator = DataCollatorWithPadding(tokenizer=tokenizer, padding = True)
	print(torch.cuda.device_count())
	if torch.cuda.device_count() > 1:
		# keeps Trainer from trying its own DataParallelism when more than 1 gpu is available
		model.is_parallelizable = True
		model.model_parallel = True
		#model.model_t5.parallelize(device_map)
	
	training_args = TrainingArguments(
			per_device_train_batch_size=batch_size,
			per_device_eval_batch_size=batch_size,
			logging_steps = 10,
			num_train_epochs=num_epoch,
			learning_rate=lr,
			optim="adamw_torch",
			evaluation_strategy="steps" if val_set_size > 0 else "no",
			save_strategy="steps",
			eval_steps= 1000 if val_set_size > 0 else None,
			save_steps = 1000,
			output_dir = output_dir,
			save_total_limit=3,
			load_best_model_at_end=True if val_set_size > 0 else False,
			#sharded_ddp = True
		)
	trainer = MultiLabelTrainer(
		model=model,
		train_dataset=train_data,
		eval_dataset=val_data,
		args = training_args,
		data_collator=DataCollatorForSeq2Seq(
			tokenizer, pad_to_multiple_of=2, return_tensors="pt", padding=True
		),
	)
	trainer.train()
	torch.save(model.state_dict(), os.path.join(output_dir, 'model_dict.pth'))

if __name__ == "__main__":
	fire.Fire(run)
