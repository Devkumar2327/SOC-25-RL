import numpy as np
from base import Agent, MultiArmedBandit
import matplotlib.pyplot as plt

class KLUCBAgent(Agent):
    def __init__(self, time_horizon, bandit: MultiArmedBandit, c=3):
        super().__init__(time_horizon, bandit)
        self.n_arms = len(bandit.arms)
        self.counts = np.zeros(self.n_arms)
        self.rewards_sum = np.zeros(self.n_arms)
        self.time_step = 0
        self.c = c

    def empirical_mean(self, arm):
        if self.counts[arm] == 0:
            return 0.0
        return self.rewards_sum[arm] / self.counts[arm]

    def kl_divergence(self, p, q):
        eps = 1e-15  # Prevents log(0)
        p = min(max(p, eps), 1 - eps)
        q = min(max(q, eps), 1 - eps)
        return p * np.log(p / q) + (1 - p) * np.log((1 - p) / (1 - q))

    def kl_ucb(self, p_hat, count, t):
        upper_bound = 1.0
        lower_bound = p_hat
        rhs = (np.log(t) + self.c * np.log(np.log(t))) / count

        for _ in range(25):  # Binary search
            mid = (upper_bound + lower_bound) / 2
            if self.kl_divergence(p_hat, mid) > rhs:
                upper_bound = mid
            else:
                lower_bound = mid
        return (upper_bound + lower_bound) / 2

    def give_pull(self):
        if self.time_step < self.n_arms:
            arm = self.time_step  # Warm-up phase: pull each arm once
        else:
            ucbs = np.zeros(self.n_arms)
            for arm in range(self.n_arms):
                p_hat = self.empirical_mean(arm)
                ucbs[arm] = self.kl_ucb(p_hat, self.counts[arm], self.time_step + 1)
            arm = np.argmax(ucbs)

        reward = self.bandit.pull(arm)
        self.reinforce(reward, arm)

    def reinforce(self, reward, arm):
        self.counts[arm] += 1
        self.rewards_sum[arm] += reward
        self.rewards.append(reward)
        self.time_step += 1

    def plot_arm_graph(self):
        counts = self.counts
        indices = np.arange(len(counts))
        plt.figure(figsize=(12, 6))
        plt.bar(indices, counts, color='lightgreen', edgecolor='black')
        plt.title('KL-UCB: Arm Pull Counts', fontsize=16)
        plt.xlabel('Arm', fontsize=14)
        plt.ylabel('Pull Count', fontsize=14)
        plt.xticks(indices, [f'Arm {i}' for i in indices])
        for i, count in enumerate(counts):
            plt.text(i, count + 0.5, str(int(count)), ha='center', va='bottom', fontsize=12)
        plt.tight_layout()
        plt.show()
