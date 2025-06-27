import numpy as np
from base import Agent, MultiArmedBandit
import matplotlib.pyplot as plt

class ThompsonSamplingAgent(Agent):
    def __init__(self, time_horizon, bandit: MultiArmedBandit):
        super().__init__(time_horizon, bandit)
        self.n_arms = len(bandit.arms)
        self.successes = np.zeros(self.n_arms)
        self.failures = np.zeros(self.n_arms)
        self.time_step = 0

    def give_pull(self):
        # Sample from Beta distribution for each arm
        samples = np.random.beta(self.successes + 1, self.failures + 1)
        arm = np.argmax(samples)

        reward = self.bandit.pull(arm)
        self.reinforce(reward, arm)

    def reinforce(self, reward, arm):
        if reward == 1:
            self.successes[arm] += 1
        else:
            self.failures[arm] += 1
        self.rewards.append(reward)
        self.time_step += 1

    def plot_arm_graph(self):
        total_pulls = self.successes + self.failures
        indices = np.arange(self.n_arms)

        plt.figure(figsize=(12, 6))
        plt.bar(indices, total_pulls, color='orange', edgecolor='black')
        plt.title('Thompson Sampling: Arm Pull Counts', fontsize=16)
        plt.xlabel('Arm', fontsize=14)
        plt.ylabel('Pull Count', fontsize=14)
        plt.xticks(indices, [f'Arm {i}' for i in indices])
        for i, count in enumerate(total_pulls):
            plt.text(i, count + 0.5, str(int(count)), ha='center', va='bottom', fontsize=12)
        plt.tight_layout()
        plt.show()
