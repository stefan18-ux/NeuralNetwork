import torch
import torch.nn.functional as F

words = open('names.txt', 'r').read().splitlines()
N = torch.zeros((27, 27), dtype = torch.int32)

chars = sorted(list(set(''.join(words))))
stoi = {s:i+1 for i,s in enumerate(chars)}
stoi['.'] = 0
itos = {i:s for s,i in stoi.items()}

for w in words:
    chs = ['.'] + list(w) + ['.']
    for ch1, ch2 in zip(chs, chs[1:]):
        i1 = stoi[ch1]
        i2 = stoi[ch2]
        N[i1, i2] += 1

g = torch.Generator().manual_seed(2147483647)
P = (N + 1).float()
P /= P.sum(1, keepdims = True)

# for i in range(5):
#     out = []
#     ix = 0
#     while True:
#         p = P[ix]
#         ix = torch.multinomial(p, num_samples=1, replacement=True, generator=g).item()
#         if ix == 0:
#             break
#         out.append(itos[ix])
#     print(''.join(out))
    

# log_likelihood = 0.0
# n = 0

# for w in words:
#   chs = ['.'] + list(w) + ['.']
#   for ch1, ch2 in zip(chs, chs[1:]):
#     ix1 = stoi[ch1]
#     ix2 = stoi[ch2]
#     prob = P[ix1, ix2]
#     logprob = torch.log(prob)
#     log_likelihood += logprob
#     n += 1
#     #print(f'{ch1}{ch2}: {prob:.4f} {logprob:.4f}')

# print(f'{log_likelihood=}')
# nll = -log_likelihood
# print(f'{nll=}')
# print(f'{nll/n}')


xs = []
ys = []

for w in words:
   chs = ['.'] + list(w) + ['.']
   for ch1, ch2 in zip(chs, chs[1:]):
      ix1 = stoi[ch1]
      ix2 = stoi[ch2]
      xs.append(ix1)
      ys.append(ix2)

xs = torch.tensor(xs)
ys = torch.tensor(ys)
num = xs.nelement()
print(f'Number of examples: {num}')

g = torch.Generator().manual_seed(2147483647)
W = torch.randn((27, 27), generator=g, requires_grad=True)
x_enc = F.one_hot(xs, num_classes=27).float()


for k in range(1000):
    # Forward pass
    logits = x_enc @ W
    counts = logits.exp()
    # Every row in the matrix contains the probabilities of the next character on the designated position
    probs = counts / counts.sum(1, keepdims=True) # Normalised probabilities
    # Loss function
    loss = -probs[torch.arange(num), ys].log().mean()

    # Backward pass
    W.grad = None
    loss.backward()
    # Tensor update
    W.data -= 50 * W.grad

print(loss.item())