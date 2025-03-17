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
minibatch_size = 32
hidden_layer_neurons = 200
vocab_size = len(itos)

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

C = torch.randn((vocab_size, emb_size), generator = g)
W1 = torch.randn(block_size * emb_size, hidden_layer_neurons, generator = g) * (5/3) / ((block_size * emb_size)**0.5)  # * 0.2
b1 = torch.randn(hidden_layer_neurons, generator = g) * 0.01
W2 = torch.randn(hidden_layer_neurons, vocab_size, generator = g) * 0.01
b2 = torch.randn(vocab_size, generator = g) * 0
# BatchNorm parameters
bngain = torch.ones((1, hidden_layer_neurons))
bnbias = torch.zeros((1, hidden_layer_neurons))
# Get a rough idea of what they are based on the training set we feed into the neural network
bnmean_running = torch.zeros((1, hidden_layer_neurons))
bnstd_running = torch.ones((1, hidden_layer_neurons))
params = [C, W1, W2, b1, b2, bngain, bnbias]
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
    ix = torch.randint(0, Xtr.shape[0], (minibatch_size,))
    emb = C[Xtr[ix]]
    embcat = emb.view(-1, block_size * emb_size)
    # Linear layer
    hpreact = embcat @ W1 + b1
    # BatchNorm layer
    # ----------------------------------------------------------------
    bnmeani = hpreact.mean(0, keepdim=True)
    bnstdi = hpreact.std(0, keepdim=True)
    hpreact = bngain * (hpreact - bnmeani) / bnstdi + bnbias

    with torch.no_grad():
        bnmean_running = 0.999 * bnmean_running + 0.001 * bnmeani
        bnstd_running = 0.999 * bnstd_running + 0.001 * bnstdi
    # ----------------------------------------------------------------
    # Non-linearity
    h = torch.tanh(hpreact)
    logits = h @ W2 + b2 #(32, 27)
    # count = logits.exp()
    # probs = count / count.sum(1, keepdims = True)
    # loss = -probs[torch.arange(32), Y].log().mean()
    loss = F.cross_entropy(logits, Ytr[ix])

    for p in params:
        p.grad = None
    loss.backward()

    lr = 0.1 if step < 100000 else 0.01
    if step > 200000:
        lr = 0.001
    for p in params:
        p.data += -lr * p.grad

    # track stats
    stepi.append(step)
    lossi.append(loss.log10().item())


# plt.hist(h.view(-1).tolist(), 50)
# plt.show()
# plt.figure(figsize=(20, 10))
# plt.imshow(h.abs() > 0.99, cmap = "grey", interpolation='nearest')

# plt.plot(stepi, lossi)
# plt.show()

print(f'Training loss is {loss.item()} from training on minibatches')

emb = C[Xtr]
embcat = emb.view(-1, block_size * emb_size)
hpreact = embcat @ W1 + b1
hpreact = bngain * (hpreact - bnmean_running) / bnstd_running + bnbias
h = torch.tanh(hpreact)
logits = h @ W2 + b2
loss = F.cross_entropy(logits, Ytr)

print(f'Training loss is {loss.item()} on the actual full tranining data')

emb = C[Xdev]
embcat = emb.view(-1, block_size * emb_size)
hpreact = embcat @ W1 + b1
hpreact = bngain * (hpreact - bnmean_running) / bnstd_running + bnbias
h = torch.tanh(hpreact)
logits = h @ W2 + b2
loss = F.cross_entropy(logits, Ydev)

print(f'Testing loss is {loss.item()}')

torch.save({
    'C': C,
    'W1': W1,
    'b1': b1,
    'W2': W2,
    'b2': b2,
    'stoi': stoi,
    'itos': itos,
    'block_size': block_size,
    'emb_size': emb_size,
    'bngain': bngain,
    'bnbias': bnbias,
    'bnmean_running': bnmean_running,
    'bnstd_running': bnstd_running
}, 'makemore.pth')