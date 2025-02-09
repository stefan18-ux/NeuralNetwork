import torch

x = torch.tensor([[1, 2], [2, 3]])
x = x.float()
x = x / x.sum(1, keepdim=True)
print(x.shape)