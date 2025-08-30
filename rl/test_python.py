import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical


class PolicyNet(nn.Module):
    def __init__(self, state_dim, n_actions, hidden=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_actions)  # outputs *logits*
        )

    def forward(self, x) -> torch.Tensor:
        return self.net(x)  # shape: [batch, n_actions]


def demo_reinforce_step():
    # --- setup ---
    torch.manual_seed(0)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    state_dim = 5
    n_actions = 3
    batch = 2

    model = PolicyNet(state_dim, n_actions).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

    # batch of states
    state = torch.randn(batch, state_dim, device=device)

    # --- standard REINFORCE using torch.distributions.Categorical ---
    logits = model(state)                 # shape [B, n_actions], requires_grad=True
    probs = F.softmax(logits, dim=-1)     # shape [B, n_actions]
    dist = Categorical(probs)

    actions = dist.sample()               # shape [B]
    logp = dist.log_prob(actions)         # shape [B]

    # assume we observed a return G per element (here example values)
    G = torch.tensor([2.0, 1.0], device=device)  # shape [B]

    # loss to minimize = - mean_i ( G_i * log pi(a_i|s_i) )
    loss = - (G * logp).mean()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print("--- After one REINFORCE update ---")
    print("actions:", actions.tolist())
    print("logp:", logp.detach().cpu().numpy())
    print("loss:", loss.item())

    # --- manual check of the derivative d log pi / d logits per batch element ---
    # Recompute logits (fresh forward pass) so gradients are with current params
    logits = model(state)                 # shape [B, n_actions]

    # compute log softmax per row
    log_soft = F.log_softmax(logits, dim=-1)  # shape [B, n_actions]

    # get the log-prob scalar for each batch element given the sampled action
    idx = torch.arange(batch, device=logits.device)  # [0,1,...,B-1]
    # actions = torch.argmax(log_soft, dim=-1)  
    logp_manuals = log_soft[idx, actions]     # shape [B]

    # grad of sum(logp_manuals) wrt logits -> yields shape [B, n_actions]
    # (since logp_i depends only on logits[i], gradient rows are independent)
    grad_logits = torch.autograd.grad(logp_manuals.sum(), logits)[0]  # shape [B, n_actions]

    probs_now = F.softmax(logits, dim=-1).detach()   # [B, n_actions], detached for comparison
    one_hot = F.one_hot(actions, num_classes=n_actions).float().to(device)  # [B, n_actions]
    expected = one_hot - probs_now                   # [B, n_actions]

    # move to cpu for nicer printing
    print("\n--- Manual gradient check (per batch row) ---")

    print(f"action={actions.tolist()}")
    print("  logits    :", logits.detach().cpu().numpy())
    print("  probs_now     :", probs_now.cpu().numpy())
    print(" logp_manuals:", logp_manuals.detach().cpu().numpy())
    print("  grad_logits:", grad_logits.detach().cpu().numpy())
    print("  one_hot  :", one_hot.cpu().numpy())
    print("  expected  :", expected.cpu().numpy())
    print("  close?    :", torch.allclose(grad_logits, expected.to(grad_logits.device), atol=1e-6))
    print("  max abs diff:", (grad_logits.detach().cpu() - expected.cpu()).abs().max().item())
    print()

    # overall check
    print("Overall close?:", torch.allclose(grad_logits, expected.to(grad_logits.device), atol=1e-6))


if __name__ == '__main__':
    demo_reinforce_step()
