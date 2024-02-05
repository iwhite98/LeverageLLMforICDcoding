#!/usr/bin/env python
# coding: utf-8

# In[1]:


import transformers_laat_kg
import transformers
import textwrap
from transformers import GenerationConfig
from transformers_laat_kg.src.transformers.models.llama.tokenization_llama import LlamaTokenizer
from transformers_laat_kg.src.transformers.models.llama.modeling_llama_kg import LlamaForCausalLM_laat_kg
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
#vocab = Vocab(data_path)
with open(OUTPUT_DIR + '/vocab.pkl', 'rb') as f:
	vocab = pickle.load(f)
	#pickle.dump(vocab, f)


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
		result = {'input_ids': result_input['input_ids'][0:possible_length-7] + result_input['input_ids'][len_input-7:len_input]+ [0] * len(result_output['input_ids'][1::]),
				 'attention_mask': result_input['attention_mask'][0:possible_length-7] + result_input['attention_mask'][len_input-7:len_input] + [1] * len(result_output['attention_mask'][1::])}
	else:
		result = {'input_ids': result_input['input_ids'][0:len_input-7] + [0]*(possible_length-len_input) + result_input['input_ids'][len_input-7:len_input] + [0] * len(result_output['input_ids'][1::]),
				 'attention_mask': result_input['attention_mask'] + [1]*(possible_length-len_input) + [1] * len(result_output['attention_mask'][1::])}



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
			result['input_ids_ref'] = result_input_ref['input_ids'][0:possible_length_ref-7] + result_input_ref['input_ids'][len_input_ref-7:len_input_ref] + [0] * len(result_output['input_ids'][1::])
			result['attention_mask_ref'] = result_input_ref['attention_mask'][0:possible_length_ref-7] + result_input_ref['attention_mask'][len_input_ref-7:len_input_ref] + [1] * len(result_output['attention_mask'][1::])
		else:
			result['input_ids_ref'] = result_input_ref['input_ids'][0:len_input_ref-7] + [0]*(possible_length_ref - len_input_ref) + result_input_ref['input_ids'][len_input_ref-7:len_input_ref] + [0] * len(result_output['input_ids'][1::])
			result['attention_mask_ref'] = result_input_ref['attention_mask'] + [1] * (possible_length_ref - len_input_ref) + [1] * len(result_output['attention_mask'][1::])
		result['input_ids_ref'].append(0)
		result['attention_mask_ref'].append(1)

		if len(result['input_ids_ref']) < max_length:
			result['input_ids_ref'] = [0] * (max_length - len(result['input_ids_ref'])) + result['input_ids_ref']
			result['attention_mask_ref'] = [0] * (max_length - len(result['input_ids_ref'])) + result['attention_mask_ref']

		result["labels_ref"] = copy.deepcopy(result["input_ids_ref"])
		result["labels_ref"][0:possible_length_ref] = [IGNORE_INDEX]*(possible_length_ref)

	result["labels"] = copy.deepcopy(result["input_ids"])
	result["labels"][0:possible_length] = [IGNORE_INDEX]*(possible_length)
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


train_data = data["train"].shuffle(seed=42)
valid_data = data["validation"].shuffle(seed=42).select(range(500))
test_data = data["test"].shuffle(seed=42).select(range(500))
train_data = train_data.map(generate_and_tokenize_prompt)
valid_data = valid_data.map(generate_and_tokenize_prompt_test)
test_data = test_data.map(generate_and_tokenize_prompt_test)


# In[8]:


# for i in range(1000):
#	  if len(train_data['input_ids'][i])!=2048 or len(train_data['labels'][i])!=2048 or len(train_data['attention_mask'][i])!=2048:
#		  print(i)


# In[9]:


LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT= 0.05
LORA_TARGET_MODULES = [
	"q_proj",
	"v_proj",
]

BATCH_SIZE = 128
MICRO_BATCH_SIZE = 4
GRADIENT_ACCUMULATION_STEPS = BATCH_SIZE // MICRO_BATCH_SIZE
LEARNING_RATE = 3e-4
TRAIN_STEPS = 800


# In[10]:


model = prepare_model_for_int8_training(model)
config = LoraConfig(
	r=LORA_R,
	lora_alpha=LORA_ALPHA,
	target_modules=LORA_TARGET_MODULES,
	lora_dropout=LORA_DROPOUT,
	bias="none",
	task_type="CAUSAL_LM",
)
model = get_peft_model(model, config)
model.print_trainable_parameters()
'''
model = PeftModel.from_pretrained(model, OUTPUT_DIR, torch_dtype = torch.float16)
print(model.state_dict()['base_model.model.ref_linear.weight'])
model.ref_linear.load_state_dict(torch.load(OUTPUT_DIR + '/kg_model.bin'))
print(model.state_dict()['base_model.model.ref_linear.weight'])
'''

# In[11]:


# from net import Net

# config = LoraConfig(
#	r=LORA_R,
#	lora_alpha=LORA_ALPHA,
#	target_modules=LORA_TARGET_MODULES,
#	lora_dropout=LORA_DROPOUT,
#	bias="none",
#	task_type="CAUSAL_LM",
# )
# model = Net(BASE_MODEL, config)


# In[12]:


training_arguments = transformers.TrainingArguments(
	per_device_train_batch_size=MICRO_BATCH_SIZE,
	gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
	warmup_steps=100,
	max_steps=TRAIN_STEPS,
	learning_rate=LEARNING_RATE,
	fp16=False,
	logging_steps=10,
	optim="adamw_torch",
	evaluation_strategy="steps",
	save_strategy="steps",
	eval_steps=30,
	save_steps=30,
	output_dir=OUTPUT_DIR,
	save_total_limit=5,
	load_best_model_at_end=True,
	report_to="tensorboard"
)



# In[13]:

from transformers_laat_kg.src.transformers.data.data_collator import DataCollatorForSeq2Seq
data_collator = DataCollatorForSeq2Seq(
	tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
)


# In[ ]:


# from trainer import MultiLabelTrainer
# 
from transformers_laat_kg.src.transformers.multilabel_trainer_kg_laat_wot_dec import Trainer
trainer = Trainer(
	model=model,
	train_dataset=train_data,
	eval_dataset=valid_data,
	args=training_arguments,
	data_collator=data_collator,
	#compute_metrics = Evaluate_Metric
)
 
model = torch.compile(model)

#trainer.train()
trainer.train(resume_from_checkpoint = OUTPUT_DIR + '/checkpoint-300')
model.save_pretrained(OUTPUT_DIR)


# In[ ]:




