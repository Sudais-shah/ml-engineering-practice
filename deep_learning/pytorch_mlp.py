import torch
import torch.nn as nn


class MultiLayerNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()  

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.fc1 = nn.Linear(self.input_dim, self.hidden_dim)
        self.act = nn.ReLU()
        self.fc2 = nn.Linear(self.hidden_dim, self.output_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x


if __name__ == "__main__":
    mlp = MultiLayerNN(784, 64, 10)
    print(mlp)

    x = torch.randn(1, 784)
    y = mlp(x)
    print(y.shape)