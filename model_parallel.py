import torch
import torch.nn as nn
import torch.optim as optim


class ToyModel(nn.Module):
    def __init__(self):
        super(ToyModel, self).__init__()
        self.net1 = torch.nn.Linear(5000, 5000).to('cuda:0')
        self.relu = torch.nn.ReLU()
        self.net2 = torch.nn.Linear(5000, 500).to('cuda:1')

    def forward(self, x):
        x = self.relu(self.net1(x.to('cuda:0')))
        x = self.net2(x.to('cuda:1'))
        return x

model = ToyModel()
loss_fn = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.001)

for i in range(30):
    optimizer.zero_grad()
    outputs = model(torch.randn(5000, 5000))
    labels = torch.randn(5000, 500).to('cuda:1')
    loss_fn(outputs, labels).backward()
    optimizer.step()