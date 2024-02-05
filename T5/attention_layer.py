"""
	This class is to implement the attention layer which supports hard attention, self-structured attention
		and self attention

	@author: Thanh Vu <thanh.vu@csiro.au>
	@date created: 20/03/2019
	@date last modified: 19/08/2020
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LabelAttentionLayer(nn.Module):

	def __init__(self,d_model, d_a, n_label, device):
		"""
		The init function
		:param args: the input parameters from commandline
		:param size: the input size of the layer, it is normally the output size of other DNN models,
			such as CNN, RNN
		"""
		super(LabelAttentionLayer, self).__init__()

		self.d_model = d_model
		self.d_a = d_a
		# For self-attention: d_a and r are the dimension of the dense layer and the number of attention-hops
		# d_a is the output size of the first linear layer

		# r is the number of attention heads

		self.n_label = n_label
		self.device = device
		self.first_linear = nn.Linear(d_a, d_model)
		self.second_linear = nn.Linear(d_model, n_label)
		self.third_linear = nn.Linear(d_a, self.n_label, bias=True)
		self._init_weights(mean=0.0, std=0.03)

	def _init_weights(self, mean=0.0, std=0.03) -> None:
		"""
		Initialise the weights
		:param mean:
		:param std:
		:return: None
		"""
		torch.nn.init.normal(self.first_linear.weight, mean, std)
		if self.first_linear.bias is not None:
			self.first_linear.bias.data.fill_(0)

		torch.nn.init.normal(self.second_linear.weight, mean, std)
		if self.second_linear.bias is not None:
			self.second_linear.bias.data.fill_(0)
		torch.nn.init.normal(self.third_linear.weight, mean, std)

	def forward(self, x):
		"""
		:param x: [batch_size x max_len x dim (i.e., self.size)]

		:param previous_level_projection: the embeddings for the previous level output
		:param label_level: the current label level
		:return:
			Weighted average output: [batch_size x dim (i.e., self.size)]
			Attention weights
		"""

		device = x.device
		self.first_linear.to(device)
		self.second_linear.to(device)
		self.third_linear.to(device)

		x = x.to(torch.float)
		weights = F.tanh(self.first_linear(x))

		att_weights = self.second_linear(weights)
		att_weights = F.softmax(att_weights, 1).transpose(1, 2)
		if len(att_weights.size()) != len(x.size()):
			att_weights = att_weights.squeeze()
		weighted_output = att_weights @ x
		weighted_output = self.third_linear.weight.mul(weighted_output).sum(dim=2).add(
				self.third_linear.bias)


		return weighted_output

	# Using when use_regularisation = True
	@staticmethod
	def l2_matrix_norm(m):
		"""
		Frobenius norm calculation
		:param m: {Variable} ||AAT - I||
		:return: regularized value
		"""
		return torch.sum(torch.sum(torch.sum(m ** 2, 1), 1) ** 0.5)
