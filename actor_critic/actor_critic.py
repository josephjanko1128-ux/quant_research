"""
Actor-Critic model that learns which single distribution has the highest mean.

Problem setup:
- There are N=5 distributions (Gaussians with different means)
- The agent selects 1 distribution per step
- Reward = sample drawn from the chosen distribution
- Goal: consistently choose the distribution with the highest mean

Architecture:
- Actor: outputs a probability vector over N distributions
- Critic: estimates the state value (expected future reward)
- Training: Advantage Actor-Critic (A2C) with entropy regularization
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── Reproducibility ─────────────────────────────────────────────────────────
torch.manual_seed(42)
np.random.seed(42)

# ── Environment ──────────────────────────────────────────────────────────────
class DistributionBanditEnv:
    """
    K-armed bandit where each arm is a Gaussian.
    Action = choose 1 arm; reward = sample from that distribution.
    """
    def __init__(self, n_dists=5):
        self.n_dists   = n_dists
        self.means     = np.array([1.0, 3.0, 5.0, 2.0, 4.0])  # optimal = idx 2
        self.stds      = np.array([4.0, 2.0, 2.0, 5.0, 2.0])
        self.n_actions = n_dists
        self.optimal   = int(np.argmax(self.means))             # idx 2, mean=5
        self.optimal_expected = self.means[self.optimal]

    def step(self, action):
        """Sample reward from the chosen distribution."""
        return float(np.random.normal(self.means[action], self.stds[action]))

    def state(self):
        """Stateless bandit -> fixed dummy state vector."""
        return torch.zeros(self.n_dists)

    def is_optimal(self, action):
        return action == self.optimal


# ── Actor-Critic Network ─────────────────────────────────────────────────────
class ActorCritic(nn.Module):
    """
    Shared body -> actor head (policy over N dists) + critic head (value).
    """
    def __init__(self, state_dim, n_actions, hidden=64):
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.actor  = nn.Linear(hidden, n_actions)
        self.critic = nn.Linear(hidden, 1)

    def forward(self, x):
        h = self.body(x)
        return self.actor(h), self.critic(h)

    def act(self, state):
        logits, value = self(state)
        probs   = F.softmax(logits, dim=-1)
        dist    = torch.distributions.Categorical(probs)
        action  = dist.sample()
        log_p   = dist.log_prob(action)
        entropy = dist.entropy()
        return action.item(), log_p, value.squeeze(), entropy


# ── Training loop ─────────────────────────────────────────────────────────────
def train(n_episodes=3000, lr=3e-3, entropy_coef=0.01):
    env   = DistributionBanditEnv(n_dists=5)
    model = ActorCritic(state_dim=env.n_dists, n_actions=env.n_actions)
    opt   = optim.Adam(model.parameters(), lr=lr)

    rewards_log = []
    optimal_log = []   # 1 if optimal distribution chosen, else 0

    window = 100

    print("=" * 60)
    print("Actor-Critic: learning the best single distribution")
    print(f"  Distribution means  : {env.means}")
    print(f"  Optimal distribution: D{env.optimal}  (mu={env.means[env.optimal]:.1f})")
    print(f"  Expected best reward: {env.optimal_expected:.2f}")
    print("=" * 60)

    for ep in range(1, n_episodes + 1):
        state = env.state()
        action, log_p, value, entropy = model.act(state)

        reward = env.step(action)
        is_opt = env.is_optimal(action)

        target    = torch.tensor(reward, dtype=torch.float32)
        advantage = (target - value).detach()

        actor_loss  = -(log_p * advantage) - entropy_coef * entropy
        critic_loss = F.mse_loss(value, target)
        loss = actor_loss + critic_loss

        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
        opt.step()

        rewards_log.append(reward)
        optimal_log.append(int(is_opt))

        if ep % 200 == 0:
            avg_r    = np.mean(rewards_log[-window:])
            opt_rate = np.mean(optimal_log[-window:]) * 100
            print(f"  Ep {ep:5d} | avg reward {avg_r:6.2f} | "
                  f"optimal-dist rate {opt_rate:5.1f}%")

    # Final greedy evaluation
    model.eval()
    with torch.no_grad():
        logits, _ = model(env.state())
        probs      = F.softmax(logits, dim=-1).numpy()
        greedy_act = int(probs.argmax())

    print("\n" + "=" * 60)
    print("Final greedy policy:")
    print(f"  Chosen dist  : D{greedy_act}  (mu={env.means[greedy_act]:.1f})")
    print(f"  True optimal : D{env.optimal}  (mu={env.means[env.optimal]:.1f})")
    print(f"  Match        : {greedy_act == env.optimal}")
    print("\nLearned probabilities per distribution:")
    for i, p in enumerate(probs):
        marker = " <- OPTIMAL" if i == env.optimal else ""
        print(f"  D{i}  mu={env.means[i]:.1f}  p={p:.4f}{marker}")

    return env, rewards_log, optimal_log, probs


# ── Plotting ─────────────────────────────────────────────────────────────────
def plot_results(env, rewards_log, optimal_log, final_probs):

    def smooth(arr, w=100):
        return np.convolve(arr, np.ones(w)/w, mode='valid')

    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    fig.suptitle("Actor-Critic: Learning the Best Single Distribution",
                 fontsize=14, fontweight='bold', y=1.01)

    colors = plt.cm.tab10(np.linspace(0, 0.9, env.n_dists))

    # 1. Distribution overview
    ax = axes[0, 0]
    x  = np.linspace(-3, 9, 400)
    for i, (mu, sigma, c) in enumerate(zip(env.means, env.stds, colors)):
        pdf = (1/(sigma * np.sqrt(2*np.pi))) * np.exp(-0.5*((x - mu)/sigma)**2)
        lw  = 2.5 if i == env.optimal else 1.2
        ls  = '-'  if i == env.optimal else '--'
        ax.plot(x, pdf, lw=lw, ls=ls, color=c,
                label=f'D{i}  mu={mu:.1f}' + (' *' if i == env.optimal else ''))
    ax.set_title("The 5 Distributions  (* = optimal)")
    ax.set_xlabel("Value"); ax.set_ylabel("Density")
    ax.legend(fontsize=9); ax.grid(alpha=0.3)

    # 2. Rolling reward
    ax = axes[0, 1]
    s_r = smooth(rewards_log)
    ax.plot(s_r, color='steelblue', lw=1.5, label='Smoothed reward (w=100)')
    ax.axhline(env.optimal_expected, color='tomato', ls='--', lw=1.4,
               label=f'Optimal expected = {env.optimal_expected:.2f}')
    ax.set_title("Episode Reward (rolling avg)")
    ax.set_xlabel("Episode"); ax.set_ylabel("Reward")
    ax.legend(fontsize=9); ax.grid(alpha=0.3)

    # 3. Optimal distribution selection rate
    ax = axes[1, 0]
    s_o = smooth(optimal_log) * 100
    ax.plot(s_o, color='mediumseagreen', lw=1.5)
    ax.axhline(100, color='gray', ls=':', lw=1)
    ax.set_ylim(0, 105)
    ax.set_title("Optimal Distribution Selection Rate (%)")
    ax.set_xlabel("Episode"); ax.set_ylabel("% choosing optimal dist")
    ax.grid(alpha=0.3)

    # 4. Final learned policy over distributions
    ax = axes[1, 1]
    bar_colors = ['tomato' if i == env.optimal else c
                  for i, c in enumerate(colors)]
    ax.bar(range(env.n_dists), final_probs, color=bar_colors)
    ax.set_xticks(range(env.n_dists))
    ax.set_xticklabels([f'D{i}\nmu={env.means[i]:.0f}' for i in range(env.n_dists)])
    ax.set_title("Final Learned Policy (action probabilities)")
    ax.set_xlabel("Distribution"); ax.set_ylabel("Probability")
    red_patch = mpatches.Patch(color='tomato', label='Optimal distribution')
    ax.legend(handles=[red_patch], fontsize=9)
    ax.grid(alpha=0.3, axis='y')

    plt.tight_layout()
    path = "./actor_critic_results.png"
    plt.savefig(path, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved -> {path}")


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    results = train(n_episodes=3000)
    plot_results(*results)
    print("\nDone.")