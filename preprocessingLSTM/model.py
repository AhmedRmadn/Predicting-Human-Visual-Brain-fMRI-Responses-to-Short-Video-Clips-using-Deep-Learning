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


# num_classes = 232
# num_epochs = 12
# batch_size = 1
# learning_rate = 0.0001

# input_size = 100
# sequence_length = 16
# hidden_size = 256
# num_layers = 6

class StatsModel(nn.Module):
    def __init__(self):
        super(StatsModel, self).__init__()

    def forward(self, seq):
        embed_min = torch.min(seq, dim=1)[0]
        embed_max = torch.max(seq, dim=1)[0]
        embed_std, embed_avg = torch.std_mean(seq, unbiased=False, dim=1)
        return torch.cat([seq[:, -1, :], embed_min, embed_max, embed_avg, embed_std], dim=1)

class RnnModel(nn.Module):
    def __init__(self, input_size, rnn_size, dropout_rate=0.4):
        super(RnnModel, self).__init__()

        self.act = nn.SiLU()
        self.dropout = nn.Dropout(dropout_rate)
        self.rnn_1 = nn.LSTM(input_size=input_size, hidden_size=rnn_size, num_layers=1, batch_first=True,)
        self.rnn_2 = nn.LSTM(input_size=input_size, hidden_size=rnn_size, num_layers=2, batch_first=True,)
        self.rnn_3 = nn.LSTM(input_size=input_size, hidden_size=rnn_size, num_layers=4, batch_first=True,)
    
    def forward(self, input):
        RNN_out_1, _ = self.rnn_1(self.dropout(input))
        RNN_out_2, _ = self.rnn_2(self.dropout(input))
        RNN_out_3, _ = self.rnn_3(self.dropout(input))

        RNN_out = torch.cat([RNN_out_1, 
                             RNN_out_2, 
                             RNN_out_3,
                             ], dim=2)

        RNN_out = RNN_out[:, -1, :]
        RNN_out = self.act(RNN_out)
        return RNN_out

class Vid2FMRIModel(nn.Module):
    def __init__(self, num_of_features,embed_size, output_size=256, rnn_features=True, dropout_rate=0.2):
        super(Vid2FMRIModel, self).__init__()
        
        self.num_of_features = num_of_features
        self.embed_size = embed_size
        self.rnn_features = rnn_features
        #self.final_embed = norm_linear(self.num_of_features, self.embed_size)

        fc_size = num_of_features*5
        self.stats = StatsModel()
        
        if self.rnn_features:
            rnn_size = 1024+256
            fc_size += rnn_size*3
            self.rnn = RnnModel(num_of_features, rnn_size)
        self.fc_size=fc_size
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = norm_linear(fc_size, embed_size)
        self.tanh = nn.Tanh()
        self.final_embed = norm_linear(self.embed_size, output_size)

    def forward(self, input):

        embeds_seq = (input)
        f_stats = self.stats(embeds_seq)
        
        if self.rnn_features:
            f_rnn = self.rnn(embeds_seq)
            f = torch.cat([f_stats, f_rnn], dim=1)
        else:
            f = torch.cat([f_stats], dim=1)

    
        f = self.dropout(f)
       
        out = self.fc(f)
        out = self.tanh(out)
        out = self.final_embed((out))

        return out
