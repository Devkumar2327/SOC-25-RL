import numpy as np
from base import Agent, MultiArmedBandit
import matplotlib.pyplot as plt

class UCBAgent(Agent):
    def __init__(self, time_horizon, bandit: MultiArmedBandit):
        super().__init__(time_horizon, bandit)
        self.n_arms = len(bandit.arms)
        self.counts = np.zeros(self.n_arms)
        self.rewards_sum = np.zeros(self.n_arms)
        self.time_step = 0

    def empirical_mean(self, arm):
        if self.counts[arm] == 0:
            return 0.0
        return self.rewards_sum[arm] / self.counts[arm]

    def ucb(self, arm, t):
        if self.counts[arm] == 0:
            return float('inf')  # Force exploration
        return self.empirical_mean(arm) + np.sqrt(2 * np.log(t) / self.counts[arm])

    def give_pull(self):
        if self.time_step < self.n_arms:
            arm = self.time_step  # Initial exploration
        else:
            ucbs = [self.ucb(arm, self.time_step + 1) for arm in range(self.n_arms)]
            arm = np.argmax(ucbs)

        reward = self.bandit.pull(arm)
        self.reinforce(reward, arm)

    def reinforce(self, reward, arm):
        self.counts[arm] += 1
        self.rewards_sum[arm] += reward
        self.rewards.append(reward)
        self.time_step += 1

    def plot_arm_graph(self):
        indices = np.arange(self.n_arms)
        plt.figure(figsize=(12, 6))
        plt.bar(indices, self.counts, color='cornflowerblue', edgecolor='black')
        plt.title('UCB1: Arm Pull Counts', fontsize=16)
        plt.xlabel('Arm', fontsize=14)
        plt.ylabel('Pull Count', fontsize=14)
        plt.xticks(indices, [f'Arm {i}' for i in indices])
        for i, count in enumerate(self.counts):
            plt.text(i, count + 0.5, str(int(count)), ha='center', va='bottom', fontsize=12)
        plt.tight_layout()
        plt.show()
