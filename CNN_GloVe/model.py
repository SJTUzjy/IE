import torch
from torch import nn

device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class MyModel(nn.Module):
    def __init__(self, in_channels=50, out_channels=200, kernel_size=3):
        super(MyModel, self).__init__()
        self.conv = nn.Sequential(
            nn.Dropout(0.2),
            nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size),   # 98 x 200
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=3),  # 32 x 200
            nn.Flatten()    # 6400
        )
        self.dense = nn.Linear(6400, 50)
        self.classify = nn.Linear(50, 1)

    def forward(self, x):
        # Input: (batch, 50, 100)
        # Output: (batch, 1)
        x = x.float()
        x = self.conv(x)
        x = self.dense(x)
        output = self.classify(x)
        return output.reshape(-1)

    def predict(self, x):
        output = self.forward(x)
        return (output - 0.5) >= 0.0