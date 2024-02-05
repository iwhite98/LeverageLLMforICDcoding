import torch.nn as nn
from transformers import AutoModel, T5ForConditionalGeneration
from attention_layer import LabelAttentionLayer
import torch
from typing import Optional, Tuple, Union
import pandas as pd
import torch.nn.functional as F

from nltk.tokenize import sent_tokenize, RegexpTokenizer
import re

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

	output= re.sub('[=+,#/\?:^.@*\"※~ㆍ!』‘|\(\)\[\]`\'…》\”\“\’·]', ' ', output)
	output = re.sub('\s+',' ', output)
	output = re.sub('--|__|==', '', output)
	output = re.sub('[\(;*?\)\[\]]', '', output)
	output = re.sub('\t', ' ', output)

	return output

class Net(nn.Module):
	def __init__(self, base_model, tokenizer, vocab):
		super(Net, self).__init__()
		#self.model_t5 = T5ForConditionalGeneration.from_pretrained(base_model, load_in_8bit = True, device_map = 'auto')
		#self.model_t5 = T5ForConditionalGeneration.from_pretrained(base_model, from_flax=True)
		self.model_t5 = T5ForConditionalGeneration.from_pretrained(base_model, device_map = 'auto')

		self.tokenizer = tokenizer
		self.vocab = vocab
		self.emb = self.model_t5.shared
		self.device = next(self.model_t5.parameters()).device
		if base_model == 't5-base':
			self.label_attention = LabelAttentionLayer(768, 768, 50, self.device)
			self.ref_linear = nn.Linear(768, 768)
		elif base_model == 't5-large' or base_model == 't5-3b':
			self.label_attention = LabelAttentionLayer(1024, 1024, 50, self.device)
			self.ref_linear = nn.Linear(1024, 1024)

		input = []
		with open('vocab_50.txt', 'r') as f:
			lines = f.readlines()
		for line in lines:
			line = line.strip()
			if '|' in line:
				continue
			input = input + line.split(' ')
		input = ' '.join(list(set(input)))

		instruction = 'The following note is the discharge summary of a patient. Provide all ICD-9 codes applicable to the patient.'

		## check oprional
		input = instruction + input
		input = clean_data(input)
		source_encoding =  tokenizer(
			input,
			max_length=2048,
			padding = 'max_length',
			return_tensors='pt',
		)
		self.input_ids_full = source_encoding['input_ids']
		self.attention_mask_full = source_encoding['attention_mask']

		three_digit = set()
		for code in self.vocab.label2index.keys():
			three_digit.add(code.split('.')[0])
		self.three_digit = list(three_digit)

	def generate(self, encoder_hidden_states, max_gen_len, temperature = 0.0, top_p = 0.95):
		bsz = len(encoder_hidden_states)
		total_len = max_gen_len

		tokens = torch.full((bsz, total_len), self.tokenizer.pad_token_id)
		tokens[:, 0] = self.model_t5.config.decoder_start_token_id
		tokens = tokens.cuda().long()


		input_text_mask = tokens != self.tokenizer.pad_token_id
		start_pos = 1
		prev_pos = 0

		for cur_pos in range(start_pos, total_len + 1):

			sequence = self.decoder_layer(decoder_input_ids = tokens[:, :cur_pos], encoder_outputs = encoder_hidden_states)

			if cur_pos == total_len:
				decoder_output = sequence
				head_output = self.lm_head_layer(sequence)
				return decoder_output, head_output

			logits = self.lm_head_layer(sequence)

			if temperature > 0:
				probs = torch.softmax(logits / temperature, dim=-1)
				next_token = self.sample_top_p(probs, top_p)
			else:
				next_token = torch.argmax(logits, dim=-1)
			
			next_token = next_token[:, -1].reshape(-1)
			
			### Early stopping
			if torch.all(next_token == 1):
				print('early stopping')
				decoder_output = sequence
				head_output = logits
				return decoder_output, head_output
			tokens[:, cur_pos] = next_token
		return logits, logits

	def generate_training(self, encoder_hidden_states, max_gen_len, temperature = 0.0, top_p = 0.95):
		bsz = len(encoder_hidden_states)
		total_len = max_gen_len

		tokens = torch.full((bsz, total_len), self.tokenizer.pad_token_id)
		tokens[:, 0] = self.model_t5.config.decoder_start_token_id
		tokens = tokens.cuda().long()
		
		head_output = torch.full((bsz, total_len), self.tokenizer.eos_token_id)

		input_text_mask = tokens != self.tokenizer.pad_token_id
		start_pos = 1
		prev_pos = 0

		for cur_pos in range(start_pos, total_len + 1):

			sequence = self.decoder_layer(decoder_input_ids = tokens[:, :cur_pos], encoder_outputs = encoder_hidden_states)

			if cur_pos == total_len:
				decoder_output = sequence
				head_output = self.lm_head_layer(sequence)
				return decoder_output, head_output

			logits = self.lm_head_layer(sequence)

			if temperature > 0:
				probs = torch.softmax(logits / temperature, dim=-1)
				next_token = self.sample_top_p(probs, top_p)
			else:
				next_token = torch.argmax(logits, dim=-1)
			
			next_token = next_token[:, -1].reshape(-1)
			
			### Early stopping
			if torch.all(next_token == 1):
				print('early stopping')
				#tokens[:, cur_pos:] = 1
				#sequence = self.decoder_layer(decoder_input_ids = tokens, encoder_outputs = encoder_hidden_states)
				decoder_output = sequence
				head_output[:, :cur_pos] = logits
				#head_output = logits
				return decoder_output, head_output

			tokens[:, cur_pos] = next_token
		return logits, logits

	def sample_top_p(self, probs, p):
		probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
		probs_sum = torch.cumsum(probs_sort, dim=-1)
		mask = probs_sum - probs_sort > p
		probs_sort[mask] = 0.0
		probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
		next_token = torch.multinomial(probs_sort, num_samples=1)
		next_token = torch.gather(probs_idx, -1, next_token)
		return next_token

	def encoder_layer(
		self,
		input_ids: Optional[torch.LongTensor] = None,
		attention_mask: Optional[torch.FloatTensor] = None,
		decoder_input_ids: Optional[torch.LongTensor] = None,
		head_mask: Optional[torch.FloatTensor] = None,
		inputs_embeds: Optional[torch.FloatTensor] = None,
		use_cache: Optional[bool] = None,
		output_attentions: Optional[bool] = None,
		output_hidden_states: Optional[bool] = None,
		return_dict: Optional[bool] = None):

		use_cache = use_cache if use_cache is not None else self.model_t5.config.use_cache
		return_dict = return_dict if return_dict is not None else self.model_t5.config.use_return_dict
	
		# FutureWarning: head_mask was separated into two input args - head_mask, decoder_head_mask
		if head_mask is not None and decoder_head_mask is None:
			if self.model_t5.config.num_layers == self.model_t5.config.num_decoder_layers:
				warnings.warn(__HEAD_MASK_WARNING_MSG, FutureWarning)
				decoder_head_mask = head_mask

		################
		'''
		device = self.model_t5.encoder.first_device
		input_ids = input_ids.to(device)
		attention_mask = attention_mask.to(device)
		'''

		encoder_outputs = self.model_t5.encoder(
			input_ids=input_ids,
			attention_mask=attention_mask,
			inputs_embeds=inputs_embeds,
			head_mask=head_mask,
			output_attentions=output_attentions,
			output_hidden_states=output_hidden_states,
			return_dict=return_dict,
		)
		return encoder_outputs

	
	def decoder_layer(
		self,
		input_ids: Optional[torch.LongTensor] = None,
		attention_mask: Optional[torch.FloatTensor] = None,
		decoder_input_ids: Optional[torch.LongTensor] = None,
		decoder_attention_mask: Optional[torch.BoolTensor] = None,
		decoder_head_mask: Optional[torch.FloatTensor] = None,
		cross_attn_head_mask: Optional[torch.Tensor] = None,
		encoder_outputs: Optional[Tuple[Tuple[torch.Tensor]]] = None,
		past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
		inputs_embeds: Optional[torch.FloatTensor] = None,
		decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
		labels: Optional[torch.LongTensor] = None,
		use_cache: Optional[bool] = None,
		output_attentions: Optional[bool] = None,
		output_hidden_states: Optional[bool] = None,
		return_dict: Optional[bool] = None,
	):	
		hidden_states = encoder_outputs

		if self.model_t5.model_parallel:
			torch.cuda.set_device(self.model_t5.decoder.first_device)
	
		if labels is not None and decoder_input_ids is None and decoder_inputs_embeds is None:
			# get decoder inputs from shifting lm labels to the right
			decoder_input_ids = self.model_t5._shift_right(labels)
	
		# Set device for model parallelism
		if self.model_t5.model_parallel:
			torch.cuda.set_device(self.model_t5.decoder.first_device)
			hidden_states = hidden_states.to(self.model_t5.decoder.first_device)
			if decoder_input_ids is not None:
				decoder_input_ids = decoder_input_ids.to(self.model_t5.decoder.first_device)
			if attention_mask is not None:
				attention_mask = attention_mask.to(self.model_t5.decoder.first_device)
			if decoder_attention_mask is not None:
				decoder_attention_mask = decoder_attention_mask.to(self.model_t5.decoder.first_device)
	
		# Decode
		decoder_outputs = self.model_t5.decoder(
			input_ids=decoder_input_ids,
			attention_mask=decoder_attention_mask,
			inputs_embeds=decoder_inputs_embeds,
			past_key_values=past_key_values,
			encoder_hidden_states=hidden_states,
			encoder_attention_mask=attention_mask,
			head_mask=decoder_head_mask,
			cross_attn_head_mask=cross_attn_head_mask,
			use_cache=use_cache,
			output_attentions=output_attentions,
			output_hidden_states=output_hidden_states,
			return_dict=return_dict,
		)
	
		sequence_output = decoder_outputs[0]
		return sequence_output
	
	def lm_head_layer(self, sequence_output):
		if self.model_t5.model_parallel:
			torch.cuda.set_device(self.model_t5.encoder.first_device)
			self.model_t5.lm_head = self.model_t5.lm_head.to(self.model_t5.encoder.first_device)
			#sequence_output = sequence_output.to(self.model_t5.encoder.first_device)
			sequence_output = sequence_output.to(self.model_t5.lm_head.weight.device)
	
		if self.model_t5.config.tie_word_embeddings:
			sequence_output = sequence_output * (self.model_t5.model_dim**-0.5)
	
		lm_logits = self.model_t5.lm_head(sequence_output)
		return lm_logits	

	def index_decode(self, batch_index):
		output = []
		for batch_idx, pred_output in enumerate(batch_index):
			pred_code = [0.0] * 50
			pred_output = pred_output.split(',')
			for code in pred_output:
				if code in self.vocab.label2index.keys():
					pred_code[self.vocab.label2index[code]] = 1
			output.append(pred_code)
		return torch.tensor(output)
	
	def forward(self, input_ids, attention_mask, decoder_labels, labels, training, input_ids_ref = None, attention_mask_ref = None):
		att_matrix = self.encoder_layer(self.input_ids_full.long(), self.attention_mask_full.long(), decoder_labels.long()).last_hidden_state ## [B, seq_len, 768]
		self.ref_linear.to(att_matrix.device)
		att_matrix = self.ref_linear(att_matrix)
		
		if training[0] :
			last_hidden_state = self.encoder_layer(input_ids.long(), attention_mask.long(), decoder_labels.long()).last_hidden_state ## [B, seq_len, 768]
			last_hidden_state = F.softmax(last_hidden_state.matmul(att_matrix.permute(0,2,1)), 1).matmul(att_matrix) + last_hidden_state
			output = self.decoder_layer(input_ids.long(), attention_mask.long(), labels = decoder_labels.long(), encoder_outputs = last_hidden_state) ## [B, seq_len, vocab_len]
			output_att = self.label_attention(last_hidden_state)
			output_decode = self.lm_head_layer(output)

			if input_ids_ref is not None:
				last_hidden_state_ref = self.encoder_layer(input_ids_ref.long(), attention_mask_ref.long(), decoder_labels.long()).last_hidden_state ## [B, seq_len, 768]
				last_hidden_state_ref = F.softmax(last_hidden_state_ref.matmul(att_matrix.permute(0,2,1)), 1).matmul(att_matrix) + last_hidden_state_ref
				output_ref = self.decoder_layer(input_ids_ref.long(), attention_mask_ref.long(), labels = decoder_labels.long(), encoder_outputs = last_hidden_state_ref) ## [B, seq_len, vocab_len]
				output_att_ref = self.label_attention(last_hidden_state_ref)
				output_decode_ref = self.lm_head_layer(output_ref)

				return output_att, output_decode, output_att_ref, output_decode_ref
				#return output_att, output_decode, output_att_ref, output_decode_ref, last_hidden_state, last_hidden_state_ref

			return output_att, output_decode
		else:
			with torch.no_grad():
				last_hidden_state = self.encoder_layer(input_ids.long(), attention_mask.long(), decoder_labels.long()).last_hidden_state ## [B, seq_len, 768]
				last_hidden_state = F.softmax(last_hidden_state.matmul(att_matrix.permute(0,2,1)), 1).matmul(att_matrix) + last_hidden_state
				#decoder_output, head_output = self.generate(last_hidden_state, max_gen_len = 512)
				output_att = self.label_attention(last_hidden_state)
			#return output_att, head_output
			return output_att, output_att
			'''			
				#return head_output
				output_dec = head_output.argmax(-1)
				output_dec = self.tokenizer.batch_decode(output_dec)
				output_dec = self.index_decode(output_dec)

			return output_att, output_dec
			'''
