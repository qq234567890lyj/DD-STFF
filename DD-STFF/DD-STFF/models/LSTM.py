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
from minepy import MINE

from scipy.signal import savgol_filter
from scipy.stats import pearsonr
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from statsmodels.tsa.stattools import grangercausalitytests
from torch import flatten, dropout
from torch.nn.functional import gelu

# from src.kan import KAN

acv = nn.GELU()




# This module is designed to perform operations of a Multi-Layer Perceptron (MLP), including linear transformation, normalization, and optional dropout, typically used for feature extraction and transformation.
# model = CNN_LSTM(in_channels=len(args.input_features), hidden_size=args.hidden_dim, num_layers=args.n_layers,out_channels=32,output_size=len(args.output_features))
class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(LSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers


        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)

        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):  # £¨368£¬19,4£©

        x = x.permute(0, 2, 1)  #


        device = x.device


        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim, device=device).requires_grad_()
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim, device=device).requires_grad_()


        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))  # out(368,19,20)


        out = self.fc(out)  # 32 512 15

        out = out.permute(0, 2, 1)  # 32 15 512

        return out

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
            LSTM(configs.enc_in+4, configs.hidden_dim,configs.num_layers,configs.c_out+4)
            for _ in range(configs.e_layers)
        ])

        # Encoder
        # self.encoder = Mixer2dTriUKAN(configs.d_model, configs.enc_in, configs.d_core, configs.grid_size,configs.hidden_dim, configs.dropout,tau=0.1, hard=True)


        # Decoder  Mapping output
        self.projection = nn.Linear(configs.d_model, configs.pred_len, bias=True)

    # x_enc£¨32,96,7 £©£¬x_mark_enc£¨32,96,4 £©
    # 32,96,137
    # 30 40 12
    def batch_high_low_freq_split(self, x, cutoff_ratio=0.2):

        fft_x = torch.fft.rfft(x, dim=2)
        freq_bins = fft_x.shape[2]
        cutoff_point = int(freq_bins * cutoff_ratio)


        low_mask = torch.zeros_like(fft_x, dtype=torch.bool)
        low_mask[:, :, :cutoff_point] = True
        high_mask = ~low_mask


        low_fft = torch.zeros_like(fft_x)
        low_fft[:, :, :cutoff_point] = fft_x[:, :, :cutoff_point]

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

        # ETT£¨32,11,512£©Value embedding
        # PEMS (32,358,512)
        # SOlar(32,137,512)
        enc_out = self.enc_embedding(x_enc, x_mark_enc)  # Embedding layer transforms data dimension to 512


        # low_time, high_time = self.batch_high_low_freq_split(enc_out)  # £¨32,11,256£©£¨32,11,256£©

        # trend, seasonal = self.decompose_etl(enc_out)

        for i in range(self.layer):
            enc_out1 = self.encoder[i](enc_out)  # enc_out(32,11,512)

            enc_out = enc_out1    # £¨32,11,1024£©


        # Restore to original input format
        # batch_size, d_series, channels = dec_out.shape
        dec_out = self.projection(enc_out).permute(0, 2, 1)[:, :, :N]  # £¨32,96,7£©
        # dec_out = self.kan(enc_out).permute(0, 2, 1)[:, :, :N]  # £¨32,96,7£©

        # De-Normalization from Non-stationary Transformer
        if self.use_norm:
            dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
            dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        return dec_out

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)  # £¨32,96,7£©
        return dec_out[:, -self.pred_len:, :]  # [B, L, D]-self.pred_len: Selects self.pred_len elements starting from the last element.








