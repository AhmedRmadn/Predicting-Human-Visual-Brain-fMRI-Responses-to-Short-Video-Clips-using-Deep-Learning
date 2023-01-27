import torch.nn as nn

def norm_linear(input_size, output_size):
    return nn.utils.weight_norm(nn.Linear(input_size, output_size))