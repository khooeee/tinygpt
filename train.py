
import torch
from torch import nn
from torch.nn import functional as F

torch.manual_seed(42)
device = "mps" if torch.backends.mps.is_available() else "cpu"

# Character tokenizer
text = open("input.txt").read()
chars = sorted(set(text))
stoi = {c: i for i, c in enumerate(chars)}
data = torch.tensor([stoi[c] for c in text], device=device)

V = len(chars)  # Vocabulary size
T = 64          # Context length
B = 16          # Batch size

# Tiny causal Transformer
class TinyGPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.token = nn.Embedding(V, 96)
        self.position = nn.Embedding(T, 96)

        layer = nn.TransformerEncoderLayer(
            d_model=96,
            nhead=4,
            dim_feedforward=384,
            dropout=0.0,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            layer, num_layers=2, enable_nested_tensor=False
        )
        self.head = nn.Linear(96, V)

    def forward(self, x):
        t = x.shape[1]
        h = self.token(x) + self.position(
            torch.arange(t, device=x.device)
        )
        mask = torch.triu(
            torch.ones(t, t, device=x.device, dtype=torch.bool),
            diagonal=1,
        )
        return self.head(self.transformer(h, mask=mask))

model = TinyGPT().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

print("Device:", device)
print("Parameters:", sum(p.numel() for p in model.parameters()))

# Training loop
for step in range(301):
    ix = torch.randint(len(data) - T - 1, (B,), device=device)
    offsets = torch.arange(T, device=device)
    x = data[ix[:, None] + offsets]
    y = data[ix[:, None] + offsets + 1]

    logits = model(x)
    loss = F.cross_entropy(logits.reshape(-1, V), y.reshape(-1))

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if step % 50 == 0:
        print(f"Step {step}: loss={loss.item():.3f}")

# Generate text
model.eval()
sequence = torch.tensor([[stoi["F"]]], device=device)

with torch.no_grad():
    for _ in range(300):
        logits = model(sequence[:, -T:])[:, -1, :]
        next_token = torch.multinomial(
            F.softmax(logits / 0.8, dim=-1), 1
        )
        sequence = torch.cat((sequence, next_token), dim=1)

print("".join(chars[i] for i in sequence[0].tolist()))
