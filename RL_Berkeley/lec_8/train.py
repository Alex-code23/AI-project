import random
import numpy as np
import time
import os
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from collections import deque

from envs import GridWorld, PointMass1D
from replay_buffer import ReplayBuffer
from q_network import MLP, QNetworkContinuous

# ---------------------------------------------------------------------
# Simple CEM optimizer for continuous action maximization
# ---------------------------------------------------------------------
def cem_maximize_q(q_net, state, action_low=-1.0, action_high=1.0,
                   iters=5, population=128, topk=10, device='cpu'):
    """
    state: torch tensor [batch, state_dim]
    returns: actions tensor [batch, 1] approximating argmax_a q_net(state,a)
    """
    batch = state.shape[0]
    mean = torch.zeros(batch, 1, device=device)
    std = torch.ones(batch, 1, device=device) * ((action_high - action_low) / 4.0)
    for _ in range(iters):
        samples = mean.unsqueeze(0) + std.unsqueeze(0) * torch.randn(population, batch, 1, device=device)
        samples = samples.clamp(action_low, action_high)
        flat_state = state.unsqueeze(0).repeat(population,1,1).reshape(population*batch, -1)
        flat_actions = samples.reshape(population*batch, 1)
        with torch.no_grad():
            vals = q_net(flat_state, flat_actions).view(population, batch)
        topk_vals, topk_idx = torch.topk(vals, topk, dim=0)
        topk_samples = []
        for b in range(batch):
            topk_samples.append(samples[topk_idx[:,b], b, 0])
        topk_samples = torch.stack(topk_samples, dim=1) # [topk, batch]
        mean = topk_samples.mean(dim=0, keepdim=True).t()
        std = topk_samples.std(dim=0, unbiased=False, keepdim=True).t() + 1e-3
    return mean.clamp(action_low, action_high)

# ---------------------------------------------------------------------
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

# ---------------------------------------------------------------------
def compute_loss(batch, net, target_net, gamma=0.99, device='cpu', double=False, continuous=False):
    states = torch.tensor(np.stack([b.state for b in batch]), dtype=torch.float32, device=device)
    actions = np.array([b.action for b in batch])
    rewards = torch.tensor(np.array([b.reward for b in batch], dtype=np.float32), device=device)
    next_states = torch.tensor(np.stack([b.next_state for b in batch]), dtype=torch.float32, device=device)
    dones = torch.tensor(np.array([b.done for b in batch], dtype=np.float32), device=device)

    if not continuous:
        q_vals = net(states)
        q_sa = q_vals.gather(1, torch.tensor(actions, dtype=torch.int64, device=device).unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            if double:
                next_q_online = net(next_states)
                next_actions = next_q_online.argmax(dim=1, keepdim=True)
                next_q_target = target_net(next_states)
                next_q = next_q_target.gather(1, next_actions).squeeze(1)
            else:
                next_q_target = target_net(next_states)
                next_q = next_q_target.max(dim=1)[0]
            target = rewards + (1.0 - dones) * gamma * next_q
        loss = nn.SmoothL1Loss()(q_sa, target)
        return loss
    else:
        actions_t = torch.tensor(actions.reshape(-1,1), dtype=torch.float32, device=device)
        q_sa = net(states, actions_t)
        with torch.no_grad():
            next_actions = cem_maximize_q(target_net, next_states, device=device)
            next_q = target_net(next_states, next_actions)
            target = rewards + (1.0 - dones) * gamma * next_q
        loss = nn.SmoothL1Loss()(q_sa, target)
        return loss

# ---------------------------------------------------------------------
def train(cfg):
    """
    cfg: dictionnaire de configuration (voir main() pour les clés)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(cfg['seed'])

    # environment selection
    if cfg['env'] == 'grid':
        env = GridWorld(size=cfg['size'], max_steps=cfg['max_steps'], one_hot=True)
        state_dim = env.n_states
        action_dim = env.n_actions
        continuous = False
    elif cfg['env'] == 'pointmass':
        env = PointMass1D(max_steps=cfg['max_steps'])
        state_dim = 2
        action_dim = 1
        continuous = True
    else:
        raise ValueError("Unknown env")

    # networks
    if not continuous:
        net = MLP(state_dim, action_dim, hidden_dims=(cfg['hidden'], cfg['hidden'])).to(device)
        target_net = MLP(state_dim, action_dim, hidden_dims=(cfg['hidden'], cfg['hidden'])).to(device)
    else:
        net = QNetworkContinuous(state_dim, action_dim=1, hidden_dims=(cfg['hidden'], cfg['hidden'])).to(device)
        target_net = QNetworkContinuous(state_dim, action_dim=1, hidden_dims=(cfg['hidden'], cfg['hidden'])).to(device)
    target_net.load_state_dict(net.state_dict())

    opt = optim.Adam(net.parameters(), lr=cfg['lr'])
    buffer = ReplayBuffer(capacity=cfg['replay_size'])

    epsilon = cfg['eps_start']
    total_steps = 0
    episode = 0
    episode_rewards = []
    losses = []
    loss_steps = []

    nstep = cfg['n_step']
    gamma = cfg['gamma']
    nstep_buffer = []

    # ---- setup matplotlib for realtime plotting ----
    plt.ion()
    fig, (ax_r, ax_l) = plt.subplots(2, 1, figsize=(8,6), sharex=True, gridspec_kw={'height_ratios':[2,1]})
    ax_r.set_title("Récompense par épisode (points) et moyenne mobile")
    ax_r.set_ylabel("Reward")
    ax_l.set_ylabel("Loss")
    ax_l.set_xlabel("Episode")
    reward_line, = ax_r.plot([], [], linestyle='-', linewidth=1)   # moving average line
    reward_scatter = ax_r.scatter([], [], s=10)                   # raw episode rewards
    loss_line, = ax_l.plot([], [], linestyle='-', linewidth=1)
    plt.tight_layout()
    ma_window = cfg.get('moving_average', 20)
    recent_rewards = deque(maxlen=ma_window)

    # training loop
    last_log_time = time.time()
    while total_steps < cfg['max_steps_total']:
        s = env.reset()
        ep_reward = 0.0
        done = False
        nstep_buffer = []
        while not done and total_steps < cfg['max_steps_total']:
            # select action
            if not continuous:
                if random.random() < epsilon:
                    a = env.sample_action()
                else:
                    with torch.no_grad():
                        s_t = torch.tensor(s, dtype=torch.float32, device=device).unsqueeze(0)
                        qvals = net(s_t)
                        a = int(qvals.argmax(dim=1).item())
                next_s, r, done, _ = env.step(a)
                nstep_buffer.append((s, a, r, next_s, done))
                # n-step aggregation
                if len(nstep_buffer) >= nstep:
                    R = 0.0
                    for i in range(nstep):
                        R += (gamma**i) * nstep_buffer[i][2]
                    s0, a0 = nstep_buffer[0][0], nstep_buffer[0][1]
                    next_state_n = nstep_buffer[-1][3]
                    done_n = nstep_buffer[-1][4]
                    buffer.push(s0, a0, R, next_state_n, done_n)
                    nstep_buffer.pop(0)
            else:
                if random.random() < epsilon:
                    a = env.sample_action()
                else:
                    s_t = torch.tensor(s, dtype=torch.float32, device=device).unsqueeze(0)
                    with torch.no_grad():
                        a_t = cem_maximize_q(net, s_t, device=device)
                        a = float(a_t.item())
                next_s, r, done, _ = env.step(a)
                buffer.push(s, float(a), r, next_s, done)

            s = next_s
            ep_reward += r
            total_steps += 1

            # training step
            if len(buffer) >= cfg['batch_size'] and total_steps % cfg['train_freq'] == 0:
                batch = buffer.sample(cfg['batch_size'])
                loss = compute_loss(batch, net, target_net, gamma=cfg['gamma'],
                                    device=device, double=(cfg['double'] and not continuous),
                                    continuous=continuous)
                opt.zero_grad()
                loss.backward()
                if cfg['grad_clip'] is not None:
                    torch.nn.utils.clip_grad_norm_(net.parameters(), cfg['grad_clip'])
                opt.step()
                losses.append(loss.item())
                loss_steps.append(episode + total_steps / max(1, cfg['max_steps_total']))

            # target network update
            if total_steps % cfg['target_update'] == 0:
                if cfg['polyak'] is None:
                    target_net.load_state_dict(net.state_dict())
                else:
                    tau = cfg['polyak']
                    for p, tp in zip(net.parameters(), target_net.parameters()):
                        tp.data.mul_(1 - tau)
                        tp.data.add_(tau * p.data)

            # epsilon decay
            if total_steps < cfg['eps_decay']:
                epsilon = cfg['eps_end'] + (cfg['eps_start'] - cfg['eps_end']) * (1 - total_steps / cfg['eps_decay'])
            else:
                epsilon = cfg['eps_end']

        # end of episode
        episode += 1
        episode_rewards.append(ep_reward)
        recent_rewards.append(ep_reward)

        # realtime plotting update every log_interval episodes (or on last episode)
        if episode % cfg['log_interval'] == 0 or total_steps >= cfg['max_steps_total']:
            # update reward scatter
            ax_r.clear()
            ax_r.set_title("Récompense par épisode (points) et moyenne mobile")
            ax_r.set_ylabel("Reward")
            ax_r.scatter(range(1, len(episode_rewards)+1), episode_rewards, s=10)
            # moving average
            if len(episode_rewards) >= 1:
                ma = []
                for i in range(len(episode_rewards)):
                    window = episode_rewards[max(0, i-ma_window+1):i+1]
                    ma.append(sum(window)/len(window))
                ax_r.plot(range(1, len(episode_rewards)+1), ma, linewidth=1)
            # show epsilon as text
            ax_r.text(0.02, 0.95, f"eps={epsilon:.3f}", transform=ax_r.transAxes)
            # loss plot
            ax_l.clear()
            ax_l.set_ylabel("Loss")
            ax_l.set_xlabel("Episode")
            if len(losses) > 0:
                # plot loss aggregated per episode roughly
                ax_l.plot(loss_steps, losses, linewidth=1)
            plt.pause(0.001)

        # logging to stdout
        if episode % cfg['log_interval'] == 0:
            avg_r = sum(episode_rewards[-cfg['log_interval']:]) / cfg['log_interval']
            print(f"[{time.strftime('%H:%M:%S')}] Episode {episode}, total_steps {total_steps}, avg_reward {avg_r:.3f}, epsilon {epsilon:.3f}")

    # end training loop
    # save final model
    os.makedirs(cfg['save_dir'], exist_ok=True)
    model_path = os.path.join(cfg['save_dir'], f"model_{cfg['env']}_{cfg['algo']}_seed{cfg['seed']}.pth")
    torch.save(net.state_dict(), model_path)
    print("Training finished. Model saved to", model_path)

    # final plot (blocking)
    plt.ioff()
    plt.show()

# ---------------------------------------------------------------------
def main():
    """
    Définir ici toutes les variables/hyperparamètres.
    Modifie ce bloc pour changer d'expérience.
    """
    cfg = {
        # environment & algo
        'env': 'grid',          # 'grid' or 'pointmass'
        'algo': 'dqn',          # 'dqn' (discrete), 'double' (discrete Double DQN), 'q_cont' (continuous Q + CEM)
        'seed': 0,

        # env settings
        'size': 6,              # grid size (only for GridWorld)
        'max_steps': 100,       # max steps per episode (env)

        # global training
        'max_steps_total': 20000,
        'replay_size': 10000,
        'batch_size': 64,
        'hidden': 128,
        'lr': 1e-3,
        'train_freq': 1,
        'target_update': 200,   # update target every N env steps
        'polyak': None,         # or small tau like 0.005 for soft updates

        # exploration / epsilon
        'eps_start': 1.0,
        'eps_end': 0.001,
        'eps_decay': 5000,

        # returns / double DQN
        'gamma': 0.99,
        'n_step': 1,
        'double': False,        # will be set to True if algo == 'double'

        # logging / misc
        'log_interval': 10,
        'grad_clip': 5.0,
        'save_dir': 'RL_Berkeley/lec_8/models',
        'moving_average': 20
    }

    # map algo flag
    if cfg['algo'] == 'double':
        cfg['double'] = True
        cfg['algo'] = 'double'
    elif cfg['algo'] == 'q_cont':
        cfg['algo'] = 'q_cont'
    else:
        cfg['algo'] = 'dqn'

    # ensure consistency: if user wants continuous algo but env is discrete, warn and adjust
    if cfg['algo'] == 'q_cont' and cfg['env'] != 'pointmass':
        print("Note: 'q_cont' is for continuous envs. Forcing env -> 'pointmass'.")
        cfg['env'] = 'pointmass'

    train(cfg)

if __name__ == "__main__":
    main()
