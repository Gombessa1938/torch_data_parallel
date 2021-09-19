import os
import sys
import tempfile
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm





class ConvNet(nn.Module):
    def __init__(self, num_classes=10):
        super(ConvNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.fc = nn.Linear(7*7*32, num_classes)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        return out

def train():
	model = ConvNet().to('cuda')

	model = nn.DataParallel(model,device_ids=[0,1,2])
	loss_fn = nn.NLLLoss(reduction='none')


	optimizer = torch.optim.SGD(model.parameters(), 1e-4)
	batch_size = 512

	train_dataset = torchvision.datasets.MNIST(root='./data',
                                               train=True,
                                               transform=transforms.ToTensor(),
                                               download=True)
	train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               num_workers=0,
                                               pin_memory=True)
    
    
    
	for epoch in tqdm(range(100)):
		for i, (images, labels) in enumerate(train_loader):
			images = images.to('cuda')
			labels = labels.to('cuda')
			optimizer.zero_grad()
			outputs = model(images)
			loss = loss_fn(outputs, labels)
			loss.mean().backward() 
			optimizer.step()
   
    

if __name__ == "__main__":
    
    n_gpus = torch.cuda.device_count()
    
    print('device count:',n_gpus)

    #assert n_gpus >= 2, f"Requires at least 2 GPUs to run, but got {n_gpus}"
    train()