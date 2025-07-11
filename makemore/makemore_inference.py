from tabnanny import check
import torch
import torch.nn.functional as F
checkpoint = torch.load('makemore.pth')

C = checkpoint['C']
W1 = checkpoint['W1']
b1 = checkpoint['b1']
W2 = checkpoint['W2']
b2 = checkpoint['b2']
stoi = checkpoint['stoi']
itos = checkpoint['itos']
block_size = checkpoint['block_size']
emb_size = checkpoint['emb_size']
bnmean_running = checkpoint['bnmean_running']
bnstd_running = checkpoint['bnstd_running']
bngain = checkpoint['bngain']
bnbias = checkpoint['bnbias']

g = torch.Generator().manual_seed(2147483647 + 10)

for _ in range(20):

    out = []
    context = [0] * block_size
    while True:
        emb = C[torch.tensor([context])]
        hpreact = emb.view(1, block_size * emb_size) @ W1 + b1
        hpreact = bngain * (hpreact - bnmean_running) / bnstd_running + bnbias
        h = torch.tanh(hpreact)
        logits = h @ W2 + b2
        probs = F.softmax(logits, dim=1)
        ix = torch.multinomial(probs, num_samples=1, generator=g).item()
        context = context[1:] + [ix]
    
        if ix == 0:
            break
        
        out.append(ix)
    
    print("".join(itos[i] for i in out))