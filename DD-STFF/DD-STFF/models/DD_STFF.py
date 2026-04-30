

import math

import numpy as np
from darts.metrics import r2_score
from einops import rearrange
from layers.Embed import DataEmbedding_inverted
from layers.KANLinear import KANLinear
from layers.Transformer_EncDec import Encoder, EncoderLayer



import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib import pyplot as plt
from minepy import MINE

from scipy.signal import savgol_filter
from scipy.stats import pearsonr
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from statsmodels.tsa.stattools import grangercausalitytests
from torch import flatten, dropout
from torch.nn.functional import gelu
from torch.nn.init import trunc_normal_

# from src.kan import KAN

acv = nn.GELU()


class NewGELU(nn.Module):
    """
    Implementation of the GELU activation function currently in Google BERT
    repo (identical to OpenAI GPT).
    Reference: Gaussian Error Linear Units (GELU) paper:
    https://arxiv.org/abs/1606.08415
    """

    def forward(self, x):
        return (
                0.5
                * x
                * (
                        1.0
                        + torch.tanh(
                    math.sqrt(2.0 / math.pi)
                    * (x + 0.044715 * torch.pow(x, 3.0))
                )
                )
        )


# Used for mixing and transforming features of input data in neural networks to better capture patterns in the data.
class TokenMixingKAN(nn.Module):
    def __init__(self, n_tokens, n_channel, n_hidden, n_hidden1=20):
        super().__init__()
        self.layer_norm = nn.LayerNorm([n_tokens, n_channel])
        self.kan1 = KANLinear(n_tokens, n_hidden)
        self.kan2 = KANLinear(n_hidden, n_tokens)
        self.activations = None

    # 32,137,128
    def forward(self, X):  # 32 11 512
        z = X.permute(0, 2, 1)
        z = self.layer_norm(z)  # LayerNorm normalizes along the last dimension (channel dimension), processing each channel independently. This keeps the shape of z as (32, 96, 11).
        z = z.permute(0, 2, 1)  # 32 11 512
        z = self.kan1(z)
        z = self.kan2(z)
        U = X + z
        self.activations = U
        return U


class ChannelMixingKAN(nn.Module):  # 32,11,512
    def __init__(self, n_tokens, n_channel, n_hidden, n_hidden1=20):
        super().__init__()
        self.layer_norm = nn.LayerNorm([n_tokens, n_channel])
        self.kan1 = KANLinear(n_channel, n_hidden)
        self.kan2 = KANLinear(n_hidden, n_channel)
        self.activations = None

    def forward(self, U):  # 32,11,512
        z = U.permute(0, 2, 1)
        z = self.layer_norm(z)  # 32,512,11
        z = self.kan1(z)
        z = self.kan2(z)
        z = z.permute(0, 2, 1)

        Y = U + z  # 32,11,512
        self.activations = Y
        return Y

#

class Mahalanobis_mask(nn.Module):
    def __init__(self, input_size,channels):  # 32 11 512
        super(Mahalanobis_mask, self).__init__()
        frequency_size = input_size // 2 + 1
        self.A = nn.Parameter(torch.randn(frequency_size, frequency_size), requires_grad=True)  # 129 129

    def calculate_prob_distance(self, X):  #32 11 512
        XF = torch.abs(torch.fft.rfft(X, dim=-1))  # 16 10 129
        X1 = XF.unsqueeze(2)  # 16 10 1 129
        X2 = XF.unsqueeze(1)  # #16 1 10 129

        # B x C x C x D
        diff = X1 - X2  # 16 10 10 129

        temp = torch.einsum("dk,bxck->bxcd", self.A, diff)  # 16 10 10 129

        dist = torch.einsum("bxcd,bxcd->bxc", temp, temp)  # 16 10 10  #

        # exp_dist = torch.exp(-dist)
        exp_dist = 1 / (dist + 1e-10)  # 16 10 10


        identity_matrices = 1 - torch.eye(exp_dist.shape[-1])  # 10 10
        mask = identity_matrices.repeat(exp_dist.shape[0], 1, 1).to(exp_dist.device)  # 32 10 10
        exp_dist = torch.einsum("bxc,bxc->bxc", exp_dist, mask)  # 16 10 10
        exp_max, _ = torch.max(exp_dist, dim=-1, keepdim=True) # 16 10 1
        exp_max = exp_max.detach()

        # B x C x C
        p = exp_dist / exp_max

        identity_matrices = torch.eye(p.shape[-1])
        p1 = torch.einsum("bxc,bxc->bxc", p, mask)  # 16 10 10

        diag = identity_matrices.repeat(p.shape[0], 1, 1).to(p.device)
        p = (p1 + diag) * 0.99  # 16 10 10

        return p




    def bernoulli_gumbel_rsample(self, distribution_matrix):
        b, c, d = distribution_matrix.shape

        flatten_matrix = rearrange(distribution_matrix, 'b c d -> (b c d) 1')
        r_flatten_matrix = 1 - flatten_matrix

        log_flatten_matrix = torch.log(flatten_matrix / r_flatten_matrix)
        log_r_flatten_matrix = torch.log(r_flatten_matrix / flatten_matrix)

        new_matrix = torch.concat([log_flatten_matrix, log_r_flatten_matrix], dim=-1)
        resample_matrix = F.gumbel_softmax(new_matrix, hard=True)

        resample_matrix = rearrange(resample_matrix[..., 0], '(b c d) -> b c d', b=b, c=c, d=d)
        return resample_matrix



    def forward(self, X):
        # p_result, exp_dist_result, dist_result = self.calculate_prob_distance(X)  #
        p_result= self.calculate_prob_distance(X)

        # bernoulli
        sample = self.bernoulli_gumbel_rsample(p_result)  # 0-1

        sample = sample*p_result  #

        return sample, p_result




class nconv(nn.Module):
    def __init__(self, gnn_type):
        super(nconv, self).__init__()
        self.gnn_type = gnn_type

    def forward(self, x, A):
        # 处理GNN类型为空间图卷积的情况
        if self.gnn_type == 'spatial':
            # x: [B, L, D], A: [B, L, L]
            # 空间图卷积: (B,L,D) × (B,L,L) -> (B,L,D)
            x = torch.einsum('bld,blw->bld', x, A)  # 修改为 bld,blw->bld
        else:
            # 默认处理时间图卷积或其他类型
            # x: [B, L, D], A: [L, L]
            x = torch.einsum('bld,lw->blw', x, A)
        return x.contiguous()


class gcn(nn.Module):
    def __init__(self, c_in, c_out, dropout, gnn_type, order=2):
        super(gcn, self).__init__()
        self.nconv = nconv(gnn_type)
        self.gnn_type = gnn_type
        self.c_in = (order + 1) * c_in
        self.mlp = nn.Linear(self.c_in, c_out)
        self.dropout = dropout
        self.order = order
        self.act = nn.GELU()

    def forward(self, x, a):
        # : x [32, 11, 512], a [32, 11, 11]
        out = [x]
        x1 = self.nconv(x, a)
        out.append(x1)

        for k in range(2, self.order + 1):
            x2 = self.nconv(x1, a)
            out.append(x2)
            x1 = x2

        h = torch.cat(out, dim=-1)   # 4 15 45 #
        h = self.mlp(h)
        h = self.act(h)
        h = F.dropout(h, self.dropout, training=self.training)
        return h




class Mixer2dTriUKAN(nn.Module):
    def __init__(self, time_steps, channels, d_core, grid_size, hidden_dim, dropout, tau=0.1, hard=True):  # 256,11
        super(Mixer2dTriUKAN, self).__init__()
        self.tau = tau
        self.hard = hard

        frequency_size = d_core // 2 + 1
        self.A = nn.Parameter(torch.randn(frequency_size, frequency_size), requires_grad=True)

        self.kan1 = KANLinear(time_steps, d_core, grid_size=grid_size)
        self.kan2 = KANLinear(d_core, time_steps, grid_size=grid_size)

        self.linear1 = nn.Linear(time_steps, d_core)
        self.linear2 = nn.Linear(d_core, time_steps)

        self.net1 = nn.Sequential(
            KANLinear(time_steps, time_steps, grid_size=grid_size),
            NewGELU(),
            KANLinear(time_steps, d_core, grid_size=grid_size),

        )
        self.net2 = nn.Sequential(
            KANLinear(d_core, time_steps, grid_size=grid_size),
            NewGELU(),
            KANLinear(time_steps, time_steps, grid_size=grid_size),
        )

        self.TokenMixingKAN = TokenMixingKAN(d_core, channels + 4, hidden_dim)

        self.TokenMixingKAN1 = TokenMixingKAN(time_steps, channels + 4, hidden_dim)


        self.ChannelMixingKAN = ChannelMixingKAN(d_core, channels + 4, hidden_dim)

        # self.TokenMixingKAN = TokenMixingKAN(d_core, channels, hidden_dim)
        #
        # self.TokenMixingKAN1 = TokenMixingKAN(time_steps, channels , hidden_dim)
        #
        # self.ChannelMixingKAN = ChannelMixingKAN(d_core, channels , hidden_dim)



        self.mask_generator = Mahalanobis_mask(d_core,channels)
        self.mask_generator1 = Mahalanobis_mask(time_steps, channels)



        self.gconv = gcn(d_core, d_core,dropout, gnn_type='spatial')
        self.gconv1 = gcn(time_steps, time_steps, dropout, gnn_type='spatial')

        self.dropout = nn.Dropout(dropout)

        # self.graph_conv = GCN(d_core)



    def forward(self, inputs, *args, **kwargs):  # (32,11,512)

        # Time dimension processing
        TokenMixing1 = self.TokenMixingKAN1(inputs)  # Extracts dependencies between different time points for

        adj1, P1 = self.mask_generator1(TokenMixing1)  # (32,11,11)  ##

        adj2 = torch.softmax(P1 , dim=-1)  #
        adj2 = self.dropout(adj2)

        # out = self.graph_conv(adj, x)+x  #

        y1 = self.gconv1(TokenMixing1, adj2)


        # batch_size, channels, d_series = inputs.shape

        combined_mean = self.kan1(inputs)  # Fully connected layer  # (32,11,512)


        # Time dimension processing
        TokenMixing = self.TokenMixingKAN(combined_mean)  # Extracts dependencies between different time points for


        adj3,P3= self.mask_generator(TokenMixing)  # (32,11,11)  ##


        adj4 = torch.softmax(P3, dim=-1)  #
        adj4 = self.dropout(adj4)

        # out = self.graph_conv(adj, x)+x  #

        y2 = self.gconv(TokenMixing ,adj4)
         # (32,11,512)


        z = self.kan2(y2)

        out = y1+z  # 32 11 512




        return out  # Returns the result of residual connection after x and channel mixing y are added. # (1026,8,5) (1026,5)





class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.batch_size = configs.batch_size
        self.layer = configs.e_layers
        # Embedding
        self.enc_embedding = DataEmbedding_inverted(configs.seq_len, configs.d_model, configs.grid_size)
        self.use_norm = configs.use_norm
        # self.stock = StockMixer(configs.batch_size, configs.d_model, market=20)

        self.encoder  = nn.ModuleList([
            Mixer2dTriUKAN(configs.d_model, configs.enc_in, configs.d_core, configs.grid_size,configs.hidden_dim, configs.dropout,tau=0.1, hard=True)
            for _ in range(configs.e_layers)
        ])

        # Encoder
        # self.encoder = Mixer2dTriUKAN(configs.d_model, configs.enc_in, configs.d_core, configs.grid_size,configs.hidden_dim, configs.dropout,tau=0.1, hard=True)


        # Decoder  Mapping output
        # self.projection = nn.Linear(configs.d_model, configs.pred_len, bias=True)
        self.projection = KANLinear(configs.d_model, configs.pred_len, grid_size=configs.grid_size)

        self.threshold_param = nn.Parameter(torch.rand(1))


    def batch_high_low_freq_split2(self, x,  adaptive_mode=True,
                                  energy_threshold=0.9):

        fft_x = torch.fft.rfft(x, dim=2)
        freq_bins = fft_x.shape[2]

        spectrum_mag = torch.abs(fft_x)

        if adaptive_mode:
            channel_energy = torch.sum(spectrum_mag ** 2, dim=2, keepdim=True)
            cumulative_energy = torch.cumsum(spectrum_mag ** 2, dim=2) / channel_energy
            mask = cumulative_energy >= energy_threshold
            cutoff_point = torch.argmax(mask.float(), dim=2, keepdim=True)
            cutoff_point = torch.max(torch.ones_like(cutoff_point), cutoff_point)
            cutoff_point = cutoff_point.expand(-1, -1, freq_bins)


        device = fft_x.device
        indices = torch.arange(freq_bins, device=device).expand_as(fft_x)
        low_mask = (indices < cutoff_point).to(fft_x.device)
        high_mask = ~low_mask

        dtype = fft_x.dtype
        low_mask = low_mask.to(dtype)
        high_mask = high_mask.to(dtype)

        low_fft = fft_x * low_mask
        high_fft = fft_x * high_mask

        low_time = torch.fft.irfft(low_fft, n=x.shape[2], dim=2)
        high_time = torch.fft.irfft(high_fft, n=x.shape[2], dim=2)

        return low_time, high_time

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):

        # Normalization from Non-stationary Transformer
        if self.use_norm:
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_enc /= stdev

        _, _, N = x_enc.shape

        enc_out = self.enc_embedding(x_enc,x_mark_enc)  # Embedding layer transforms data dimension to 512


        low_time,  high_time = self.batch_high_low_freq_split2(enc_out)  # （32,11,256）（32,11,256）


        # trend, seasonal = self.decompose_etl(enc_out)

        for i in range(self.layer):
            enc_out1 = self.encoder[i](low_time)  # enc_out(32,11,512)

            enc_out2 = self.encoder[i](high_time)
            enc_out = enc_out1 + enc_out2  #（32,11,1024）


        # Restore to original input format
        # batch_size, d_series, channels = dec_out.shape
        dec_out = self.projection(enc_out).permute(0, 2, 1)[:, :, :N]  # （32,96,7）
        # dec_out = self.kan(enc_out).permute(0, 2, 1)[:, :, :N]  # （32,96,7）

        # De-Normalization from Non-stationary Transformer
        if self.use_norm:
            dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
            dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        return dec_out

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)  # （32,96,7）
        return dec_out[:, -self.pred_len:, :]  # [B, L, D]-self.pred_len: Selects self.pred_len elements starting from the last element.



