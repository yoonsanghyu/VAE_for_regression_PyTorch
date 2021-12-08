import torch
import torch.nn as nn

class RMSELoss(nn.Module):
	def __init__(self, eps=1e-6, reduction="mean"):
		super().__init__()
		self.mse = nn.MSELoss(reduction=reduction)
		self.eps = eps

	def forward(self, y_hat, y):
		return torch.sqrt(self.mse(y_hat, y)+self.eps)