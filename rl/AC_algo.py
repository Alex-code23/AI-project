"""
DDPG-like minimal without gym + visualizations (matplotlib subplots) — English plots.

Environment:
 - state : [position, velocity, target_position]
 - action : continuous acceleration in [-1, 1]
 - objective : reach target_position (reward = -distance - action_cost)
Plots:
 - rewards per episode + moving average
 - histogram of actions taken during training
 - critic / actor losses (per update)
 - minimum distance per episode + moving success rate (moving average)
"""

import math
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# -----------------------
# Hyperparameters
# -----------------------
SEED = 42
MAX_EPISODES = 400
MAX_STEPS = 250 
BATCH_SIZE = 128
BUFFER_CAPACITY = 100000
GAMMA = 0.99
TAU = 0.001
ACTOR_LR = 1e-4
CRITIC_LR = 1e-4
START_STEPS = 100  # random steps at start
NOISE_SCALE = 0.01
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
np.random.seed(SEED)
random.seed(SEED)
torch.manual_seed(SEED)

# -----------------------
# Simple point environment
# -----------------------
class SimplePointEnv:
    def __init__(self, dt=0.1, max_acc=1.0, max_pos=5.0, goal_threshold=0.1):
        self.dt = dt
        self.max_acc = max_acc
        self.max_pos = max_pos
        self.goal_threshold = goal_threshold
        self.action_low = -1.0
        self.action_high = 1.0
        self.obs_dim = 3  # pos, vel, goal_pos
        self.act_dim = 1
        self._state = None
        self.max_steps = MAX_STEPS
        self._steps = 0

    def reset(self):
        pos = np.random.uniform(-self.max_pos/3, self.max_pos/3)
        vel = 0.0
        target = np.random.uniform(-self.max_pos, self.max_pos)
        self._state = np.array([pos, vel, target], dtype=np.float32)
        self._steps = 0
        return self._state.copy()

    def step(self, action):
        # get acc
        a = float(np.clip(action, self.action_low, self.action_high))
        # get current state
        pos, vel, target = self._state
        # some physics
        vel = vel + a * self.max_acc * self.dt
        pos = pos + vel * self.dt
        pos = np.clip(pos, -self.max_pos * 2.0, self.max_pos * 2.0)
        self._state = np.array([pos, vel, target], dtype=np.float32)
        self._steps += 1

        dist = abs(pos - target)
        reward = np.exp(-5.0 * dist) - 1 * abs(a)**2  # reward in [~0,1] for small dist
    

        done = False
        reached = False
        if dist < self.goal_threshold:
            reward += 50.0
            done = True
            reached = True
        if self._steps >= self.max_steps:
            reward -= 30
            done = True

        info = {"distance": dist, "reached": reached}
        return self._state.copy(), float(reward), done, info

    def sample_action(self):
        # for greeedy exploration
        return np.array([np.random.uniform(self.action_low, self.action_high)], dtype=np.float32)


# -----------------------
# Replay Buffer
# -----------------------
class ReplayBuffer:
    def __init__(self, obs_dim, act_dim, capacity=100000):
        self.capacity = capacity
        self.obs_buf = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.next_obs_buf = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros((capacity, act_dim), dtype=np.float32)      # action
        self.rew_buf = np.zeros((capacity, 1), dtype=np.float32)            # reward
        self.done_buf = np.zeros((capacity, 1), dtype=np.float32)           # done
        self.ptr = 0            # pointeur
        self.size = 0           # actual size of our buffers

    def add(self, obs, act, rew, next_obs, done):
        # we delete previous one if it existed
        idx = self.ptr % self.capacity
        self.obs_buf[idx] = obs
        self.act_buf[idx] = act
        self.rew_buf[idx] = rew
        self.next_obs_buf[idx] = next_obs
        self.done_buf[idx] = float(done)
        self.ptr += 1
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        # choosing our sample of BATCH SIZE
        idxs = np.random.randint(0, self.size, size=batch_size)
        batch = dict(
            obs = torch.as_tensor(self.obs_buf[idxs], device=DEVICE),
            acts = torch.as_tensor(self.act_buf[idxs], device=DEVICE),
            rews = torch.as_tensor(self.rew_buf[idxs], device=DEVICE),
            next_obs = torch.as_tensor(self.next_obs_buf[idxs], device=DEVICE),
            done = torch.as_tensor(self.done_buf[idxs], device=DEVICE),
        )
        return batch



# -----------------------
# Networks (Actor / Critic)
# -----------------------
class Actor(nn.Module):
    """
    Actor = le conducteur : il choisit l'action à faire dans chaque état (la politique).
    L'actor choisit une action a dans l'état obs.
    """
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.LayerNorm(128),   # normalisation des features
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.LayerNorm(128),   # 2ème normalisation
            nn.ReLU(),
            nn.Linear(128, act_dim),
            nn.Tanh()            # borne les actions entre -1 et 1
        )

    def forward(self, x):
        return self.net(x)



class Critic(nn.Module):
    """
    Critic = l'entraîneur / juge : il évalue combien cette action est bonne (estime la valeur ou le Q-value)
    """
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim + act_dim, 128),
            nn.LayerNorm(128),   # normalisation des features
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.LayerNorm(128),   # 2ème normalisation
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, obs, act):
        x = torch.cat([obs, act], dim=-1)
        return self.net(x)


def soft_update(target, source, tau):
    """
    Mettre à jour doucement les paramètres du réseau.
    Améliore la convergence
    """
    for t_param, s_param in zip(target.parameters(), source.parameters()):
        t_param.data.copy_(t_param.data * (1.0 - tau) + s_param.data * tau)


# -----------------------
# Training loop (with metrics)
# -----------------------
def train():
    env = SimplePointEnv()
    obs_dim = env.obs_dim
    act_dim = env.act_dim

    actor = Actor(obs_dim, act_dim).to(DEVICE)
    critic = Critic(obs_dim, act_dim).to(DEVICE)
    actor_target = Actor(obs_dim, act_dim).to(DEVICE)
    critic_target = Critic(obs_dim, act_dim).to(DEVICE)
    actor_target.load_state_dict(actor.state_dict())
    critic_target.load_state_dict(critic.state_dict())

    actor_opt = optim.Adam(actor.parameters(), lr=ACTOR_LR)
    critic_opt = optim.Adam(critic.parameters(), lr=CRITIC_LR)

    buffer = ReplayBuffer(obs_dim, act_dim, capacity=BUFFER_CAPACITY)
    total_steps = 0

    # metrics to collect
    rewards_history = []
    action_history = []
    actor_losses = []
    critic_losses = []
    min_distances = []
    success_flags = []
    number_steps = []

    for ep in range(1, MAX_EPISODES + 1):
        obs = env.reset()               # reset env to start from zero
        ep_reward = 0.0                 # reward to 0
        min_dist = float("inf")
        reached_flag = 0

        for step in range(env.max_steps):
            total_steps += 1
            if total_steps < START_STEPS:
                # random action
                action = env.sample_action()
                
            else:
                obs_t = torch.as_tensor(obs, dtype=torch.float32, device=DEVICE).unsqueeze(0)
                with torch.no_grad():
                    action = actor(obs_t).cpu().numpy()[0]
                action = action + np.random.normal(scale=NOISE_SCALE, size=act_dim)
                action = np.clip(action, env.action_low, env.action_high)

            action_history.append(float(action[0]))

            next_obs, reward, done, info = env.step(action)

            min_dist = min(min_dist, info.get("distance", np.inf))
            if info.get("reached", False):
                reached_flag = 1

            buffer.add(obs, action, reward, next_obs, done)
            obs = next_obs
            ep_reward += reward

            # update networks
            if buffer.size >= BATCH_SIZE:
                batch = buffer.sample(BATCH_SIZE)
                obs_b = batch["obs"]
                acts_b = batch["acts"]
                rews_b = batch["rews"]
                next_obs_b = batch["next_obs"]
                done_b = batch["done"]

                # critic update
                with torch.no_grad():
                    # Compute the next action using the *target actor*
                    next_actions = actor_target(next_obs_b)
                    # The target critic estimates Q(s', a') for next states and actions
                    q_next = critic_target(next_obs_b, next_actions)
                    # Build the Bellman target:
                    # Q_target = r + γ * (1 - done) * Q_target(s', a')
                    # -> if done=1, we cut off the future return
                    q_target = rews_b + GAMMA * (1.0 - done_b) * q_next

                # Current Q(s,a) estimated by the *online critic*
                q_val = critic(obs_b, acts_b)

                # Critic loss = mean squared error between current Q and target Q
                critic_loss = nn.functional.mse_loss(q_val, q_target)
                critic_opt.zero_grad()
                critic_loss.backward()
                torch.nn.utils.clip_grad_norm_(critic.parameters(), 1.0)
                critic_opt.step()

                # actor update
                actor_opt.zero_grad()
                cur_actions = actor(obs_b)
                actor_loss = -critic(obs_b, cur_actions).mean()
                actor_loss.backward()
                torch.nn.utils.clip_grad_norm_(actor.parameters(), 1.0)
                actor_opt.step()

                # soft update for target models
                soft_update(critic_target, critic, TAU)
                soft_update(actor_target, actor, TAU)

                critic_losses.append(float(critic_loss.item()))
                actor_losses.append(float(actor_loss.item()))

            if done:
                number_steps.append(env._steps)
                break

        rewards_history.append(ep_reward)
        min_distances.append(min_dist)
        success_flags.append(reached_flag)

        if ep % 20 == 0 or ep == 1:
            avg_last = np.mean(rewards_history[-10:])
            succ_rate_10 = np.mean(success_flags[-10:]) if len(success_flags) >= 1 else 0.0
            print(f"Episode {ep:3d} | ep_reward {ep_reward:6.2f} | avg_last10 {avg_last:6.2f} | succ10 {succ_rate_10:.2f} | buffer {buffer.size}")

    print("Training finished.")
    metrics = {
        "rewards": np.array(rewards_history),
        "actions": np.array(action_history),
        "actor_losses": np.array(actor_losses),
        "critic_losses": np.array(critic_losses),
        "min_distances": np.array(min_distances),
        "success_flags": np.array(success_flags),
        "number_steps": np.array(number_steps)
    }
    return actor, critic, metrics, env


# -----------------------
# Plot utilities (English labels)
# -----------------------
def moving_average(x, window):
    if len(x) < 1:
        return np.array([])
    window = max(1, int(window))
    return np.convolve(x, np.ones(window)/window, mode='valid')


def plot_metrics(metrics):
    rewards = metrics["rewards"]
    actions = metrics["actions"]
    actor_losses = metrics["actor_losses"]
    critic_losses = metrics["critic_losses"]
    min_distances = metrics["min_distances"]
    success_flags = metrics["success_flags"]
    number_steps = metrics["number_steps"]

    fig, axes = plt.subplots(3, 2, figsize=(12, 8))

    # 1) Rewards per episode + moving average
    ax = axes[0, 0]
    ax.plot(rewards, label="Reward per episode")
    ma = moving_average(rewards, window=10)
    if ma.size > 0:
        ax.plot(np.arange(len(ma)) + (10 - 1), ma, label="Moving average (10)", linewidth=2)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Reward")
    ax.set_title("Reward per episode")
    ax.legend()
    ax.grid(True)

    # 2) Histogram / action distribution
    ax = axes[0, 1]
    if actions.size > 0:
        ax.hist(actions, bins=100, density=True, alpha=0.7)
    ax.set_xlabel("Action (acceleration)")
    ax.set_ylabel("Density")
    ax.set_title(f"Action distribution (n={len(actions)})")
    ax.grid(True)

    # 3) Critic / Actor losses (per update)
    ax = axes[1, 0]
    if critic_losses.size > 0:
        ax.plot(critic_losses, label="Critic loss")
        ax.plot(moving_average(critic_losses, window=max(1, int(len(critic_losses)/50))), label="Critic MA", linewidth=2)
    ax.set_xlabel("Update index")
    ax.set_ylabel("Loss")
    ax.set_title("Losses during training Critic")
    ax.legend()
    ax.grid(True)

    # 4) Min distance per episode + moving success rate
    ax = axes[1, 1]
    ax.plot(min_distances, label="Min distance per episode")
    ma_dist = moving_average(min_distances, window=10)
    if ma_dist.size > 0:
        ax.plot(np.arange(len(ma_dist)) + (10 - 1), ma_dist, label="MA distance (10)", linewidth=2)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Distance")
    ax.set_title("Minimum distance reached per episode")
    ax.grid(True)

    # twin axis for success rate
    ax2 = ax.twinx()
    succ_ma = moving_average(success_flags, window=10)
    if succ_ma.size > 0:
        ax2.plot(np.arange(len(succ_ma)) + (10 - 1), succ_ma, linestyle='--', label='Success rate MA(10)')
    ax2.set_ylabel("Success rate (MA10)")
    ax2.set_ylim(-0.05, 1.05)

    # actor losses
    ax = axes[2,0]
    if actor_losses.size > 0:
        ax.plot(-actor_losses, label="Actor loss")
        ax.plot(moving_average(-actor_losses, window=max(1, int(len(actor_losses)/50))), label="Actor MA", linewidth=2)
    ax.set_xlabel("Update index")
    ax.set_ylabel(" -Loss")
    ax.set_title("Losses during training Actor")
    ax.legend()
    ax.grid(True)

    # number of steps do you need
    ax = axes[2,1]
    ax.plot(number_steps, label="number_steps")
    steps_needed = 45
    ax.axhline(y=steps_needed, c="r")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Number steps")
    ax.set_title("Number Steps per episode")



    plt.tight_layout()
    plt.show()


# -----------------------
# Demo & run
# -----------------------
def demo(actor, env, episodes=5):
    for ep in range(episodes):
        s = env.reset()
        ep_r = 0.0
        traj = []
        for t in range(env.max_steps):
            s_t = torch.as_tensor(s, dtype=torch.float32, device=DEVICE).unsqueeze(0)
            with torch.no_grad():
                a = actor(s_t).cpu().numpy()[0]
            a = np.clip(a, env.action_low, env.action_high)
            s, r, done, info = env.step(a)
            ep_r += r
            traj.append((float(s[0]), float(s[1])))  # pos, vel
            if done:
                break
        print(f"Demo Ep {ep+1} | reward {ep_r:6.2f} | steps {len(traj)}")
        print("positions (first 10):", [round(p, 2) for p, v in traj[:10]], "...")


if __name__ == "__main__":
    trained_actor, trained_critic, metrics, env = train()
    print("\n--- Demo of trained policy ---")
    demo(trained_actor, env, episodes=3)
    print("\n--- Showing metrics ---")
    plot_metrics(metrics)
