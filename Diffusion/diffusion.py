import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

# --- Config rapide (modifie si tu veux) ---
torch.manual_seed(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
out_path = "Diffusion/diffusion_demo.png"
image_size = 64
N = 1024           # nombre d'images synthétiques
T = 500           # nombre d'étapes de diffusion (pour embedding, etc.)
batch_size = 64
n_steps = 600      # itérations d'entraînement (court pour démo)

# --- utilitaires : images synthétiques (blobs) ---
def make_blob_image(size=64):
    img = np.zeros((size, size), dtype=np.float32)
    nb = np.random.randint(1, 3)
    for _ in range(nb):
        x0 = np.random.uniform(0, size)
        y0 = np.random.uniform(0, size)
        sigma = np.random.uniform(size*0.04, size*0.12)
        amplitude = np.random.uniform(0.6, 1.0)
        xs = np.arange(size)
        ys = np.arange(size)
        xv, yv = np.meshgrid(xs, ys)
        blob = amplitude * np.exp(-((xv-x0)**2 + (yv-y0)**2) / (2*sigma**2))
        img += blob
    img = img / (img.max() + 1e-9)
    img = img * 2 - 1   # dans [-1, 1]
    return img.astype(np.float32)

# dataset en mémoire
data = [torch.from_numpy(make_blob_image(image_size)).unsqueeze(0) for _ in range(N)]

# --- schedule DDPM lite ---
betas = np.linspace(1e-4, 0.02, T, dtype=np.float32) # beta
alphas = 1.0 - betas                                  
alpha_cumprod = np.cumprod(alphas, axis=0) # cumprod = cumulative product of elements along a given axis.
sqrt_alpha_cumprod = np.sqrt(alpha_cumprod)
sqrt_one_minus_alpha_cumprod = np.sqrt(1 - alpha_cumprod)

def extract(arr, t, shape):
    out = torch.from_numpy(arr).to(device)[t].float()
    return out.view(-1, *([1]*(len(shape)-1)))

def q_sample(x0, t, noise=None):
    if noise is None:
        noise = torch.randn_like(x0)
    sqrt_ab = extract(sqrt_alpha_cumprod, t, x0.shape)
    sqrt_1_ab = extract(sqrt_one_minus_alpha_cumprod, t, x0.shape)
    return sqrt_ab * x0 + sqrt_1_ab * noise

# embedding sinusoïdal pour timestep
def timestep_embedding(timesteps, dim):
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(0, half, dtype=torch.float32) / half).to(device)
    args = timesteps.float().unsqueeze(1) * freqs.unsqueeze(0)
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    if dim % 2 == 1:
        emb = F.pad(emb, (0,1,0,0))
    return emb

# --- petit réseau qui prédit l'epsilon ---
class SmallDiffNet(nn.Module):
    def __init__(self, emb_dim=64):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv4 = nn.Conv2d(128, 128, 3, padding=1)
        self.conv5 = nn.Conv2d(128, 64, 3, padding=1)
        self.conv6 = nn.Conv2d(64, 32, 3, padding=1)
        self.out = nn.Conv2d(32, 1, 3, padding=1)
        self.time_mlp = nn.Sequential(
            nn.Linear(emb_dim, emb_dim),
            nn.ReLU(),
            nn.Linear(emb_dim, 128)
        )
        self.emb_dim = emb_dim
    def forward(self, x, t):
        temb = timestep_embedding(t, self.emb_dim)
        temb = self.time_mlp(temb).unsqueeze(-1).unsqueeze(-1)
        h = F.relu(self.conv1(x))
        h = F.relu(self.conv2(h))
        h = F.relu(self.conv3(h))
        h = h + temb
        h = F.relu(self.conv4(h))
        h = F.relu(self.conv5(h))
        h = F.relu(self.conv6(h))
        eps = self.out(h)
        return eps

# --- initialisation modèle ---
model = SmallDiffNet(emb_dim=32).to(device)
opt = torch.optim.Adam(model.parameters(), lr=3e-4)
loss_fn = nn.MSELoss()

# --- court entraînement demo ---
for step in range(1, n_steps+1):
    idx = np.random.choice(N, batch_size, replace=False)
    xb = torch.stack([data[i] for i in idx]).to(device)   # Bx1xHxW
    B = xb.shape[0] # B = batch size
    t = torch.randint(0, T, (B,), device=device, dtype=torch.long)
    eps = torch.randn_like(xb)
    x_noisy = q_sample(xb, t, eps)
    eps_pred = model(x_noisy, t)
    loss = loss_fn(eps_pred, eps)
    opt.zero_grad(); loss.backward(); opt.step()
    if step % 10 == 0:
        print(f"step {step}/{n_steps} loss {loss.item():.6f}")

# --- inference sur une image (démonstration visuelle) ---
model.eval()
with torch.no_grad():
    x0 = data[10].to(device).unsqueeze(0)        # 1x1xHxW
    t_demo = torch.tensor([120], device=device)  # étape "assez bruyante"
    noise = torch.randn_like(x0)
    x_t = q_sample(x0, t_demo, noise)
    eps_pred = model(x_t, t_demo)
    sqrt_ab_t = float(sqrt_alpha_cumprod[t_demo.item()])
    sqrt_1_ab_t = float(sqrt_one_minus_alpha_cumprod[t_demo.item()])
    x0_pred = (x_t - sqrt_1_ab_t * eps_pred) / (sqrt_ab_t + 1e-8)
    # convertir en numpy et concaténer pour affichage
    x0_np = x0.squeeze().cpu().numpy()
    x_t_np = x_t.squeeze().cpu().numpy()
    x0_pred_np = x0_pred.squeeze().cpu().numpy()
    comp = np.concatenate([x0_np, x_t_np, x0_pred_np], axis=1)  # H x (3W)
    plt.imsave(out_path, comp)
    print(f"Saved demo image to {out_path}")
