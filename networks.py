import torch
import torch.nn as nn
import numpy as np

class RNNOptimizer(nn.Module):
	def __init__(self, preproc=False, hidden_sz=20, preproc_factor=10.0, use_second_layer=False,
	n_features=4, no_tanh=False, learnable_scale=False):
		super().__init__()
		self.n_features = n_features
		self.hidden_sz = hidden_sz
		if preproc:
			self.recurs = nn.LSTMCell(2 * self.n_features, hidden_sz)
		else:
			self.recurs = nn.LSTMCell(self.n_features, hidden_sz)
		self.recurs2 = nn.LSTMCell(hidden_sz, hidden_sz)
		self.output = nn.Linear(hidden_sz, 3, bias=False)
		if no_tanh:
			self.act = lambda x: x
		else:
			self.act = lambda x: torch.tanh(x)
		self.preproc = preproc
		self.preproc_factor = preproc_factor
		self.preproc_threshold = np.exp(-preproc_factor)
		self.use_second_layer = use_second_layer
	def forward(self, inp, hidden, cell):
		if self.preproc:
			inp = inp.data
			inp2 = torch.zeros(inp.size()[0], self.n_features * 2, device=inp.data.device)
			keep_grads = (torch.abs(inp) >= self.preproc_threshold).squeeze()
			inp2[:, 0:self.n_features][keep_grads] = (torch.log(torch.abs(inp[keep_grads]) + 1e-8) / self.preproc_factor).squeeze()
			inp2[:, self.n_features:2*self.n_features][keep_grads] = torch.sign(inp[keep_grads]).squeeze()
			inp2[:, 0:self.n_features][~keep_grads] = -1
			inp2[:, self.n_features:2*self.n_features][~keep_grads] = (float(np.exp(self.preproc_factor)) * inp[~keep_grads]).squeeze()
			inp = inp2
		# print(inp.shape)
		hidden0, cell0 = self.recurs(inp, (hidden[0], cell[0]))
		if self.use_second_layer:
			hidden1, cell1 = self.recurs2(hidden0, (hidden[1], cell[1]))
			output = self.output(hidden1)
			act = self.act(output[:, :3])
			return act, [1, 1], (hidden0, hidden1), (cell0, cell1)
		else:
			output = self.output(hidden0)
			# output = self.act(output[:, :3])
			act = self.act(output[:, :3])
			return act, [1, 1], (hidden0, ), (cell0, )