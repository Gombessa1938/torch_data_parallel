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


def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12345'

    # initialize the process group
    dist.init_process_group("gloo", rank=rank, world_size=world_size)  #gloo for windoes, NCCL for linux, single node multi GPU best performance is NCCL 

def cleanup():
    dist.destroy_process_group()

def run_demo(demo_fn, world_size):
    
    mp.spawn(demo_fn,
             args=(world_size,),
             nprocs=world_size,
             join=True)





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


def train(rank, world_size):


    setup(rank, world_size)


    model = ConvNet().to(rank)
    ddp_model = DDP(model,device_ids= [rank])
    
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
            images = images.to(rank)
            labels = labels.to(rank)
        

            optimizer.zero_grad()
            outputs = model(images)
            loss = loss_fn(outputs, labels)
            loss.mean().backward() 
            optimizer.step()
   
    cleanup()

if __name__ == "__main__":
    
    n_gpus = torch.cuda.device_count()
    
    print('device count:',n_gpus)

    assert n_gpus >= 2, f"Requires at least 2 GPUs to run, but got {n_gpus}"

    world_size = n_gpus
    
    run_demo(train, world_size)
