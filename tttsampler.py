import torch
from torch.utils.data import DataLoader, DistributedSampler, Dataset

class Toydataset(Dataset):
    def __init__(self, size:int=250) -> None:
        super().__init__()
        self.data = torch.rand(size,32,32)
        self.label = torch.arange(0,size)
    def __getitem__(self, index):
        return self.data[index], self.label[index]
    def __len__(self):
        return len(self.label)

dataset = Toydataset(size=250)
sampler = DistributedSampler(dataset,num_replicas=5)
loader = DataLoader(dataset, shuffle=(sampler is None),
                    sampler=sampler)
for epoch in range(start_epoch, n_epochs):
    if is_distributed:
        sampler.set_epoch(epoch)
    train(loader)