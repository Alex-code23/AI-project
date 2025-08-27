import torch
import torch.nn as nn
import torch.optim as optim

class LowLevelRNN(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.rnn = nn.RNNCell(dim, dim)
    def forward(self, zL, zH, x):
        return self.rnn(zL + zH + x)

class HighLevelRNN(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.rnn = nn.RNNCell(dim, dim)
    def forward(self, zH, zL):
        return self.rnn(zH + zL)

class HRM(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.L = LowLevelRNN(dim)
        self.H = HighLevelRNN(dim)
        self.output_head = nn.Linear(dim, 2)
    
    def forward_cycle(self, x, zH, zL, T=3):
        # Fast updates for L, slow update for H
        for t in range(T-1):
            with torch.no_grad():  # No gradient for intermediate L
                zL = self.L(zL, zH, x)
        # Last step: allow gradients (1-step gradient)
        zL = self.L(zL, zH, x)
        zH = self.H(zH, zL)
        y_hat = self.output_head(zH)
        return zH, zL, y_hat

if __name__ == "__main__":
    # Training loop with deep supervision
    model = HRM(dim=16)
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()


    # Larger dummy batch (batch size 32)
    batch_size = 32
    X = torch.rand((batch_size, 16))
    y = torch.randint(0, 2, (batch_size,))


    zH = torch.zeros(batch_size, 16)
    zL = torch.zeros(batch_size, 16)


    N_cycles = 2
    for epoch in range(20):
        optimizer.zero_grad()
        for cycle in range(N_cycles):
            zH, zL, y_hat = model.forward_cycle(X, zH, zL, T=3)
            loss = criterion(y_hat, y)
            # Detach hidden states to prevent full BPTT
            zH = zH.detach()
            zL = zL.detach()
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")
