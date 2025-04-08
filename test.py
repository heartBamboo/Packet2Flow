# import torch
#
# from models.bert_pytorch.model import BERT
#
# input = torch.randn(32, 20, 768)
# bert_model = BERT(hidden=768, n_layers=12, attn_heads=12, dropout=0.1)
# output = bert_model(input,0.3)
# print(bert_model)
# print(output.size())
# import numpy as np
# import torch

# print(torch.__version__)
# print(torch.cuda.is_available())
# print(torch.version.cuda)
#
# data  = np.load('./y_train.npy')
# print("Original dataset labels: ", np.bincount(data))
# print(data.shape())

import torch
from mamba.mamba_ssm import Mamba

batch, length, dim = 2, 64, 16
x = torch.randn(batch, length, dim).to("cuda")
model = Mamba(
    # This module uses roughly 3 * expand * d_model^2 parameters
    d_model=dim, # Model dimension d_model
    d_state=16,  # SSM state expansion factor
    d_conv=4,    # Local convolution width
    expand=2,    # Block expansion factor
).to("cuda")
y = model(x)
assert y.shape == x.shape
