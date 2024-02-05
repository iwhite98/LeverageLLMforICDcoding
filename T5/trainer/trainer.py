from torch import nn
import torch
from transformers import Trainer
import pickle
from compute_metric import *

loss_att_log = []
loss_dec_log = []
loss_att_ref_log = []
loss_dec_ref_log = []

class MultiLabelTrainer(Trainer):
	def compute_loss(self, model, inputs, return_outputs=False):
		labels = inputs.get("labels")
		decoder_input_ids = inputs.get('decoder_input_ids')
		decoder_labels = inputs.get('decoder_labels')
		input_ids = inputs.get('input_ids')
		attention_mask = inputs.get('attention_mask')
		training = inputs.get('training')
		input_ids_ref = inputs.get("input_ids_ref")
		attention_mask_ref = inputs.get("attention_mask_ref")

		if training[0]:
			BCE_fn = nn.BCEWithLogitsLoss()
			CE_fn = nn.CrossEntropyLoss(ignore_index = -100)
			MSE_fn = nn.MSELoss()
	
			if input_ids_ref is not None:
				output_att, output_decoder, output_ref_att, output_ref_decoder = model(input_ids, attention_mask, decoder_labels, labels, training, input_ids_ref, attention_mask_ref)
				labels = labels.to(output_att.device)
				decoder_labels = decoder_labels.to(output_decoder.device)
	
				loss_att = BCE_fn(output_att, labels)
				loss_dec = CE_fn(output_decoder.view(-1, output_decoder.size(-1)), decoder_labels.view(-1))
				loss_att_ref = BCE_fn(output_ref_att, labels)
				loss_dec_ref = CE_fn(output_ref_decoder.view(-1, output_ref_decoder.size(-1)), decoder_labels.view(-1))
				loss_cr = MSE_fn(torch.sigmoid(output_att), torch.sigmoid(output_ref_att))
	
				loss_ = loss_att.to(loss_dec.device) + loss_dec
				loss_ref = loss_att_ref.to(loss_dec.device) + loss_dec_ref
				loss = 0.7 * loss_ + 0.3 * loss_ref + 0.3 * loss_cr.to(loss_dec.device)
				#loss.requires_grad = True
				
	
				return (loss, output_decoder) if return_outputs else loss
			else:
				output_att, output_decoder = model(input_ids, attention_mask, decoder_labels, labels, training, None, None)
				labels = labels.to(output_att.device)
				decoder_labels = decoder_labels.to(output_decoder.device)

				loss_att = BCE_fn(output_att, labels)
				loss_dec = CE_fn(output_decoder.view(-1, output_decoder.size(-1)), decoder_labels.view(-1))
				loss =  loss_att.to(loss_dec.device) + loss_dec
				return (loss, output_decoder) if return_outputs else loss
		else:
			output_att, output_decoder = model(input_ids, attention_mask, decoder_labels, labels, training, None, None)
			#BCE_fn = nn.BCEWithLogitsLoss()
			#BCE_fn_ = nn.BCELoss()

			f1_fn = micro_f1
			loss = f1_fn(labels.cpu().numpy(), np.rint(torch.sigmoid(output_att).cpu().numpy()))[-1]
			#loss = auc_fn(labels.cpu().numpy(), output_decoder.cpu().numpy(), 'micro')
			loss = 1 - torch.tensor(loss).to(output_decoder.device)
	
			return (loss, output_decoder) if return_outputs else loss


