#!/usr/bin/env python
# coding: utf-8

# In[1]:


import transformers_laat_kg
import transformers
import textwrap
from transformers import GenerationConfig
from transformers_laat_kg.src.transformers.models.llama.tokenization_llama import LlamaTokenizer
from transformers_laat_kg.src.transformers.models.llama.modeling_llama import LlamaForCausalLM_laat_kg
import os
import sys
from typing import List
 
from peft import (
	LoraConfig,
	get_peft_model,
	get_peft_model_state_dict,
	prepare_model_for_int8_training,
	PeftModel
)

import fire
import torch
from datasets import load_dataset
import pandas as pd
 
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
from pylab import rcParams
import numpy as np
import json
import copy
import pickle
from vocab import Vocab
from compute_metric import *
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEVICE

IGNORE_INDEX = -100


# In[2]:


# df_train = pd.read_csv("50/train.csv")
# df_valid = pd.read_csv("50/valid.csv")
# df_test = pd.read_csv("50/test.csv")
# df_train.head()


# In[3]:


# dataset_train = [
#	  {
#		  "instruction": row_dict["instruction"],#'The following note is the discharge summary of a patient. Provide all ICD-9 codes applicable to the patient.',
#		  "input": row_dict["input"],
#		  "output": row_dict["output"]
#	  }
#	  for row_dict in df_train.to_dict(orient="records")
# ]
# dataset_valid = [
#	  {
#		  "instruction": row_dict["instruction"],#'The following note is the discharge summary of a patient. Provide all ICD-9 codes applicable to the patient.',
#		  "input": row_dict["input"],
#		  "output": row_dict["output"]
#	  }
#	  for row_dict in df_valid.to_dict(orient="records")
# ]
# dataset_test = [
#	  {
#		  "instruction": row_dict["instruction"],#'The following note is the discharge summary of a patient. Provide all ICD-9 codes applicable to the patient.',
#		  "input": row_dict["input"],
#		  "output": row_dict["output"]
#	  }
#	  for row_dict in df_test.to_dict(orient="records")
# ]

# with open("50/train.json", "w") as f:
#	 json.dump(dataset_train, f)

# with open("50/valid.json", "w") as f:
#	 json.dump(dataset_valid, f)

# with open("50/test.json", "w") as f:
#	 json.dump(dataset_test, f)


# In[4]:


BASE_MODEL = "decapoda-research/llama-7b-hf"
 
model = LlamaForCausalLM_laat_kg.from_pretrained(
	BASE_MODEL,
	load_in_8bit=True,
	torch_dtype=torch.float16,
	device_map="auto",
)
 
tokenizer = LlamaTokenizer.from_pretrained(BASE_MODEL)
 
tokenizer.pad_token_id = (
	0  # unk. we want this to be different from the eos token
)
tokenizer.padding_side = "left"


# In[5]:

data = load_dataset('json', data_files={'train': './code_gpt_as_refer_reordering_chunk_json/50/train.json', 
									   'validation': './code_gpt_as_refer_reordering_chunk_json/50/valid.json',
									   'test': './code_gpt_as_refer_reordering_chunk_json/50/test.json'})

OUTPUT_DIR = "medalpaca_code_laat_wot_dec_kg_vocab_reordering_course"
data_path = './code/50'
with open(OUTPUT_DIR + '/vocab.pkl', 'rb') as f:
	vocab = pickle.load(f)


# In[6]:

### For vocab attention layer

instruction = f'The following note is the discharge summary of a patient. Provide all ICD-9 codes applicable to the patient.'
input = []

with open('vocab_50.txt', 'r') as f:
	lines = f.readlines()
for line in lines:
	line = line.strip()
	if '|' in line:
		continue
	input = input + line.split(' ')
input_att = ' '.join(list(set(input)))

def generate_prompt(data_point):
	temp1 = f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.
### Instruction:
{data_point["instruction"]}
### Input:
{data_point["input"]}
### Response:
"""
	temp1_ = f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.
### Instruction:
{data_point["instruction"]}
### Input:
{data_point["input_ref"]}
### Response:
"""
	temp2 = f"""{data_point["output"]}"""
	return temp1, temp1_, temp2

input_prompt_att, _,  __ =  generate_prompt({'instruction':instruction, 'input': input_att, 'input_ref' : '', 'output' : ''})

def tokenize(input_prompt, input_prompt_ref, response_prompt, add_eos_token=True, training = True):
	result_input = tokenizer(
		input_prompt,
		truncation=False,
#		  max_length=9999,#CUTOFF_LEN
		padding=False,
		return_tensors=None,
	)

	
	result_input_att = tokenizer(
		input_prompt_att,
		truncation=False,
#		  max_length=9999,#CUTOFF_LEN
		padding=False,
		return_tensors=None,
	)

	result_output = tokenizer(
		response_prompt,
		truncation=False,
#		  max_length=9999,
		padding=False,
		return_tensors=None,
	)
	max_length = 2048
	
	len_input = len(result_input['input_ids'])
	len_input_att = len(result_input_att['input_ids'])
	len_output = len(result_output['input_ids'])
	
	possible_length = max_length-len_output
	
	if len_input > possible_length:
		result = {'input_ids': result_input['input_ids'][0:possible_length-7] + result_input['input_ids'][len_input-7:len_input],
				 'attention_mask': result_input['attention_mask'][0:possible_length-7] + result_input['attention_mask'][len_input-7:len_input]}
	else:
		result = {'input_ids': result_input['input_ids'][0:len_input-7] + [0]*(possible_length-len_input) + result_input['input_ids'][len_input-7:len_input],
				 'attention_mask': result_input['attention_mask'] + [1]*(possible_length-len_input)}



	possible_length_att = max_length
	
	if len_input_att > possible_length_att:
		result['input_ids_att'] = result_input_att['input_ids'][0:possible_length_att-7] + result_input_att['input_ids'][len_input_att-7:len_input_att]
		result['attention_mask_att'] = result_input_att['attention_mask'][0:possible_length_att-7] + result_input_att['attention_mask'][len_input_att-7:len_input_att]
	else:
		result['input_ids_att'] = result_input_att['input_ids'][0:len_input_att-7] + [0]*(possible_length_att - len_input_att) + result_input_att['input_ids'][len_input_att-7:len_input_att][1::]
		result['attention_mask_att'] = result_input_att['attention_mask'] + [1] * (possible_length_att - len_input_att)

	result['input_ids'].append(0)
	result['input_ids_att'].append(0)
	result['attention_mask'].append(1)
	result['attention_mask_att'].append(1)


	if training:
		result_input_ref = tokenizer(
			input_prompt_ref,
			truncation=False,
	#		  max_length=9999,#CUTOFF_LEN
			padding=False,
			return_tensors=None,
		)
		len_input_ref = len(result_input_ref['input_ids'])
		possible_length_ref = max_length-len_output
		
		if len_input_ref > possible_length_ref:
			result['input_ids_ref'] = result_input_ref['input_ids'][0:possible_length_ref-7] + result_input_ref['input_ids'][len_input_ref-7:len_input_ref] + result_output['input_ids'][1::]
			result['attention_mask_ref'] = result_input_ref['attention_mask'][0:possible_length_ref-7] + result_input_ref['attention_mask'][len_input_ref-7:len_input_ref] + result_output['attention_mask'][1::]
		else:
			result['input_ids_ref'] = result_input_ref['input_ids'][0:len_input_ref-7] + [0]*(possible_length_ref - len_input_ref) + result_input_ref['input_ids'][len_input_ref-7:len_input_ref] + result_output['input_ids'][1::]
			result['attention_mask_ref'] = result_input_ref['attention_mask'] + [1] * (possible_length_ref - len_input_ref) + result_output['attention_mask'][1::]
		result['input_ids_ref'].append(0)
		result['attention_mask_ref'].append(1)

		if len(result['input_ids_ref']) < max_length:
			result['input_ids_ref'] = [0] * (max_length - len(result['input_ids_ref'])) + result['input_ids_ref']
			result['attention_mask_ref'] = [0] * (max_length - len(result['input_ids_ref'])) + result['attention_mask_ref']
		else:
			result["labels_ref"] = copy.deepcopy(result["input_ids_ref"])
			result["labels_ref"][0:possible_length_ref] = [IGNORE_INDEX]*(possible_length_ref)

	#result["labels"] = copy.deepcopy(result["input_ids"])
	#result["labels"][0:possible_length] = [IGNORE_INDEX]*(possible_length)
	result['code_labels'] = index_decode(response_prompt).squeeze()
	return result
 
def generate_and_tokenize_prompt(data_point):
	input_prompt, input_prompt_ref, response_prompt = generate_prompt(data_point)
	tokenized_full_prompt = tokenize(input_prompt, input_prompt_ref, response_prompt, training = True)
	return tokenized_full_prompt

def generate_and_tokenize_prompt_test(data_point):
	input_prompt, input_prompt_ref, response_prompt = generate_prompt(data_point)
	tokenized_full_prompt = tokenize(input_prompt, input_prompt_ref, response_prompt, training = False)
	return tokenized_full_prompt

def index_decode(test):
	true_code = [0.0] * 50
	for code in test.split(','):
		if code in vocab.label2index.keys():
			true_code[vocab.label2index[code]] = 1
	return torch.tensor(true_code)


# In[7]:


test_data = data["test"]
test_data = test_data.map(generate_and_tokenize_prompt_test)



model = PeftModel.from_pretrained(model, OUTPUT_DIR, torch_dtype = torch.float16)
model.ref_linear.load_state_dict(torch.load(OUTPUT_DIR + '/kg_model.bin'))
model.label_attention.load_state_dict(torch.load(OUTPUT_DIR + '/laat_model.bin'))

def generate(input_ids, temperature = 0, top_p = 0):

	bsz = len(input_ids)


	min_prompt_size = min([len(t) for t in input_ids])

	total_len = 2048
	tokens = torch.full((bsz, total_len), tokenizer.pad_token_id).cuda().long()

	for k, t in enumerate(input_ids):
		tokens[k, : len(t)] = torch.tensor(t).long()

	input_text_mask = tokens != tokenizer.pad_token_id
	start_pos = min_prompt_size
	for cur_pos in range(start_pos, total_len):
		output = model(input_ids = tokens[:, :cur_pos], input_ids_att = input_ids_att)
		logits = output.logits
		if cur_pos == start_pos:
			output_att = torch.sigmoid(output.output_att)
			return None, output_att

		if temperature > 0:
			probs = torch.softmax(logits / temperature, dim=-1)
			next_token = sample_top_p(probs, top_p)
		else:
			next_token = torch.argmax(logits, dim=-1)

		next_token = next_token[:, -1].reshape(-1)
		# only replace token if prompt has already been generated
		next_token = torch.where(
			input_text_mask[:, cur_pos], tokens[:, cur_pos], next_token
		)
		tokens[:, cur_pos] = next_token
	
	decoded = tokenizer.batch_decode(tokens)
	return decoded, output_att

def generate(input_ids, input_ids_att, temperature = 0, top_p = 0):

	bsz = len(input_ids)


	min_prompt_size = min([len(t) for t in input_ids])

	total_len = 2048
	tokens = torch.full((bsz, total_len), tokenizer.pad_token_id).cuda().long()

	for k, t in enumerate(input_ids):
		tokens[k, : len(t)] = torch.tensor(t).long()

	input_text_mask = tokens != tokenizer.pad_token_id
	start_pos = min_prompt_size
	for cur_pos in range(start_pos, total_len):
		output = model(input_ids = tokens[:, :cur_pos], input_ids_att = input_ids_att)
		logits = output.logits
		if cur_pos == start_pos:
			output_att = torch.sigmoid(output.output_att)
			return None, output_att

		if temperature > 0:
			probs = torch.softmax(logits / temperature, dim=-1)
			next_token = sample_top_p(probs, top_p)
		else:
			next_token = torch.argmax(logits, dim=-1)

		next_token = next_token[:, -1].reshape(-1)
		# only replace token if prompt has already been generated
		next_token = torch.where(
			input_text_mask[:, cur_pos], tokens[:, cur_pos], next_token
		)
		tokens[:, cur_pos] = next_token

	decoded = tokenizer.batch_decode(tokens)
	return decoded


def sample_top_p(probs, p):
	probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
	probs_sum = torch.cumsum(probs_sort, dim=-1)
	mask = probs_sum - probs_sort > p
	probs_sort[mask] = 0.0
	probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
	next_token = torch.multinomial(probs_sort, num_samples=1)
	next_token = torch.gather(probs_idx, -1, next_token)
	return next_token


print(len(test_data))
result = dict()
from tqdm import tqdm
true_label = []
pred_label = []
with torch.no_grad():
	for idx, data in tqdm(enumerate(test_data)):
		input_ids = torch.tensor(data['input_ids']).unsqueeze(0).to('cuda')
		input_ids_att = torch.tensor(data['input_ids_att']).unsqueeze(0).to(input_ids.device)
		output, output_att = generate(input_ids, input_ids_att)
		true_label.append(data['code_labels'])
		pred_label.append(output_att.squeeze().detach().cpu().tolist())
	print(calculate_eval_metrics(true_label, pred_label, True))
	with open(OUTPUT_DIR + '/true_label.pkl', 'wb') as f:
		pickle.dump(true_label, f)
	
	with open(OUTPUT_DIR + '/pred_label.pkl', 'wb') as f:
		pickle.dump(pred_label, f)
