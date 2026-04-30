import math

import torch
import torch.nn as nn
from layers.KANLinear import KANLinear



class DataEmbedding_inverted(nn.Module):
    def __init__(self, c_in, d_model, grid_size,dropout=0.1):
        super(DataEmbedding_inverted, self).__init__()
        self.value_embedding = nn.Linear(c_in, d_model)
        # self.value_embedding = KANLinear(c_in,  d_model, grid_size=grid_size)
        self.dropout = nn.Dropout(p=dropout)
        # Positional embedding
        self.position_embedding = PositionalEmbedding(d_model)
# 32,96,137
    def forward(self, x, x_mark):
        x = x.permute(0, 2, 1)  # (32,7,96)
        # x: [Batch Variate Time]
        if x_mark is None:
            x = self.value_embedding(x)
        else:
            # (32, 11, 96)，11 = 7 + 4。
            # the potential to take covariates (e.g. timestamps) as tokens
            x = self.value_embedding(torch.cat([x, x_mark.permute(0, 2, 1)], 1))
            # x = self.value_embedding(torch.cat([x, x_mark.permute(0, 2, 1)], 1))+self.position_embedding(torch.cat([x, x_mark.permute(0, 2, 1)], 1))  # (32,11,512)
        # x: [Batch Variate d_model]
        return self.dropout(x)


class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):  # 32,11,512
        super(PositionalEmbedding, self).__init__()
        # Compute the positional encodings once in log space.
        # Create a positional encoding matrix. Here, a zero matrix of size (max_len, d_model)
        # named 'pe' is created, and its data type is set to float. require_grad = False indicates
        # that this matrix does not require gradients, meaning it will not be updated during backpropagation.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        # Generate position indices
        # This line generates a sequence from 0 to max_len-1, converts it to float, and adds a dimension
        # using unsqueeze(1), making its shape (max_len, 1).
        position = torch.arange(0, max_len).float().unsqueeze(1)
        # Calculate the division term
        # Generate a sequence of even numbers from 0 to d_model-1, convert it to float, and compute
        # -(math.log(10000.0) / d_model) multiplied by this sequence, then take the exponent.
        # This div_term is used to generate the sine and cosine terms in the positional encoding.
        div_term = (torch.arange(0, d_model, 2).float()
                    * -(math.log(10000.0) / d_model)).exp()
        # Fill the even and odd columns of the positional encoding matrix
        # 0::2 means starting from the 0th column, filling every 2 columns.
        # 1::2 means starting from the 1st column, filling every 2 columns.
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # Add a dimension and register as a buffer
        # Here, a dimension is added, making the shape of pe (1, max_len, d_model), and it is registered as a buffer.
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        # Using register_buffer, pe is registered as a buffer, so it will be automatically saved and restored
        # when the model is loaded, but it will not be updated by the optimizer.

    def forward(self, x):
        return self.pe[:, :x.size(1)]


class PatchEmbedding(nn.Module):
    def __init__(self, d_model, patch_len, stride, padding, dropout):
        super(PatchEmbedding, self).__init__()
        # Patching
        self.patch_len = patch_len
        self.stride = stride
        self.padding_patch_layer = nn.ReplicationPad1d((0, padding))

        # Backbone, Input encoding: projection of feature vectors onto a d-dim vector space
        self.value_embedding = nn.Linear(patch_len, d_model, bias=False)

        # Positional embedding
        self.position_embedding = PositionalEmbedding(d_model)

        # Residual dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):  # 32,7,96
        # do patching
        n_vars = x.shape[1]  # 7
        x = self.padding_patch_layer(x)  # 32,7,104  使用 padding_patch_layer 对输入 x 进行填充。
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        x = torch.reshape(x, (x.shape[0] * x.shape[1], x.shape[2], x.shape[3]))  # 224,12,16
        # Input encoding
        x = self.value_embedding(x) + self.position_embedding(x)  # 224,12,512
        return self.dropout(x), n_vars