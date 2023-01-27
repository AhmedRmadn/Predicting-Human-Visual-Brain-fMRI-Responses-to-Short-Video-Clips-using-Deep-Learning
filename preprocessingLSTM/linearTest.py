from cProfile import label
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import numpy as np
import os
import glob
from utils.helper import save_dict,load_dict, saveasnii
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from layer import norm_linear

class StatsModel(nn.Module):
    def __init__(self):
        super(StatsModel, self).__init__()

    def forward(self, seq):
        embed_min = torch.min(seq, dim=1)[0]
        embed_max = torch.max(seq, dim=1)[0]
        embed_std, embed_avg = torch.std_mean(seq, unbiased=False, dim=1)
        return torch.cat([seq[:, -1, :], embed_min, embed_max, embed_avg, embed_std], dim=1)

class Vid2FMRIModel(nn.Module):
    def __init__(self, num_of_features,embed_size, output_size=256, rnn_features=True, dropout_rate=0.2):
        super(Vid2FMRIModel, self).__init__()
        
        self.num_of_features = num_of_features
        self.embed_size = embed_size
        self.rnn_features = rnn_features
        self.final_embed = norm_linear(self.num_of_features, self.embed_size)

        fc_size = embed_size
  

        self.relu = nn.ReLU()

        self.dropout = nn.Dropout(dropout_rate)
        self.fc = norm_linear(fc_size, output_size)

    def forward(self, input):
        embeds_seq = self.final_embed(input)
       
        f = self.relu(embeds_seq)
        f = self.dropout(f)
        out = self.fc(f)
        return out
