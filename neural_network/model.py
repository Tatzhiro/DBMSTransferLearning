import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class VectorPredictor(nn.Module):
    """
    input: database metrics
    output: important parameter vector, normalized
    """
    def __init__(self, input_size, output_size, num_hidden_layer=5, hidden_size=128):
        super(VectorPredictor, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.hidden_layers = nn.ModuleList([nn.Linear(hidden_size, hidden_size) for i in range(num_hidden_layer)])
        self.output = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        for layer in self.hidden_layers:
            x = layer(x)
            x = F.relu(x)
        x = self.output(x)
        x = F.softmax(x, dim=1)
        return x
