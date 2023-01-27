import torch


class WeightedMSELoss(torch.nn.Module):
    def __init__(self, reduction="mean"):
        super(WeightedMSELoss, self).__init__()
        self.reduction = reduction

    def forward(self, input, target, weight):
        WSE = weight * (input - target) ** 2
        if self.reduction == "mean":
            return torch.mean(WSE)
        else:
            return WSE
