import torch.nn as nn

class AlignLayer(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(AlignLayer, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        x = self.linear(x)
        return x