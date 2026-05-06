import torch

class SpectraModel(torch.nn.Module):
    def __init__(self):
        super(SpectraModel, self).__init__()
        self.linear1 = torch.nn.Linear(2048, 1024)
        self.act1 = torch.nn.ReLU()
        self.linear2 = torch.nn.Linear(1024, 512)
        self.act2 = torch.nn.ReLU()
        self.out = torch.nn.Linear(512, 698) 

    def forward(self, x):
        x = self.linear1(x)
        x = self.act1(x)
        x = self.linear2(x)
        x = self.act2(x)
        x = self.out(x)
        return x 
