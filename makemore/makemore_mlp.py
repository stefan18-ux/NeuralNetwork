from numpy import linspace
import torch
import torch.nn.functional as F
import random
import matplotlib.pyplot as plt


# Just mapping
words = open("names.txt", 'r').read().splitlines()
chars = sorted(list(set(''.join(words))))
stoi = {s:i+1 for i,s in enumerate(chars)}
stoi['.'] = 0
itos = {i:s for s,i in stoi.items()}

block_size = 3
emb_size = 15

def build_dataset(words):
    X, Y = [], []

    for w in words:
        context = [0] * block_size
        for ch in w + '.':
            ix = stoi[ch]
            X.append(context)
            Y.append(ix)
            context = context[1:] + [ix]

    X = torch.tensor(X)
    Y = torch.tensor(Y)

    print(X.shape, Y.shape)
    return X, Y

random.seed(42)
random.shuffle(words)
n1 = int(0.8 * len(words))
n2 = int(0.9 * len(words))

Xtr, Ytr = build_dataset(words[:n1])
Xdev, Ydev = build_dataset(words[n1:n2])
Xte, Yte = build_dataset(words[n2:])

g = torch.Generator().manual_seed(2147483647)


C = torch.randn((27, emb_size), generator = g)
W1 = torch.randn(block_size * emb_size, 300, generator = g)
b1 = torch.randn(300, generator = g)
W2 = torch.randn(300, 27, generator = g)
b2 = torch.randn(27, generator = g)
params = [C, W1, W2, b1, b2]
# print(sum(p.nelement() for p in params))

for p in params:
    p.requires_grad = True


lre = linspace(-3, 0, 1000)
lrs = 10**lre

steps = 200000
lri = []
stepi = []
lossi = []

for step in range(steps):
    # minibatch construct 
    ix = torch.randint(0, Xtr.shape[0], (1000,))
    emb = C[Xtr[ix]]
    h = torch.tanh(emb.view(-1, block_size * emb_size) @ W1 + b1)
    logits = h @ W2 + b2 #(32, 27)
    # count = logits.exp()
    # probs = count / count.sum(1, keepdims = True)
    # loss = -probs[torch.arange(32), Y].log().mean()
    loss = F.cross_entropy(logits, Ytr[ix])

    for p in params:
        p.grad = None
    loss.backward()

    lr = 0.1 if step < 100000 else 0.01
    for p in params:
        p.data += -lr * p.grad

    # track stats
    stepi.append(step)
    lossi.append(loss.log10().item())


plt.plot(stepi, lossi)
plt.show()

print(f'Training loss is {loss.item()} from training on minibatches')

emb = C[Xdev]
h = torch.tanh(emb.view(-1, block_size * emb_size) @ W1 + b1)
logits = h @ W2 + b2
loss = F.cross_entropy(logits, Ydev)

print(f'Testing loss is {loss.item()}')


emb = C[Xtr]
h = torch.tanh(emb.view(-1, block_size * emb_size) @ W1 + b1)
logits = h @ W2 + b2
loss = F.cross_entropy(logits, Ytr)

print(f'Training loss is {loss.item()} on the actual full tranining data')

