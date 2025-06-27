import numpy as np
import matplotlib.pyplot as plt
from base import MultiArmedBandit
from epsilon_greedy import EpsilonGreedyAgent
from klucb import KLUCBAgent
from ucb import UCBAgent
from thompson import ThompsonSamplingAgent

TIME_HORIZON = 30000

def run_agent(agent_class, bandit_probs, **kwargs):
    bandit = MultiArmedBandit(np.array(bandit_probs))
    agent = agent_class(TIME_HORIZON, bandit, **kwargs)
    for _ in range(TIME_HORIZON):
        agent.give_pull()
    return agent, bandit

def plot_reward_regret_curves(agents, labels):
    plt.figure(figsize=(12, 5))

    # Cumulative Reward Curve
    plt.subplot(1, 2, 1)
    for agent, label in zip(agents, labels):
        plt.plot(np.cumsum(agent.rewards[:TIME_HORIZON]), label=label)
    plt.title("Cumulative Reward")
    plt.xlabel("Time Step")
    plt.ylabel("Reward")
    plt.legend()

    # Cumulative Regret Curve
    plt.subplot(1, 2, 2)
    for agent, label in zip(agents, labels):
        optimal_reward = max(agent.bandit.arms) * np.arange(1, TIME_HORIZON + 1)
        rewards_cumsum = np.cumsum(agent.rewards[:TIME_HORIZON])
        regret = optimal_reward - rewards_cumsum
        plt.plot(regret, label=label)
    plt.title("Cumulative Regret")
    plt.xlabel("Time Step")
    plt.ylabel("Regret")
    plt.legend()

    plt.tight_layout()
    plt.show()

def plot_optimal_arm_counts(agents, labels, optimal_arm):
    counts = []
    for agent in agents:
        if hasattr(agent, "get_arm_counts"):
            counts.append(agent.get_arm_counts()[optimal_arm])
        else:
            if hasattr(agent, "count_memory"):
                counts.append(agent.count_memory[optimal_arm])
            elif hasattr(agent, "counts"):
                counts.append(agent.counts[optimal_arm])
            elif hasattr(agent, "successes") and hasattr(agent, "failures"):
                counts.append(agent.successes[optimal_arm] + agent.failures[optimal_arm])
            else:
                counts.append(0)

    plt.figure(figsize=(8, 5))
    plt.bar(labels, counts, color='teal')
    plt.title("Times Optimal Arm Was Pulled")
    plt.ylabel("Count")
    for i, count in enumerate(counts):
        plt.text(i, count + 300, f"{int(count)}", ha='center', fontsize=12)
    plt.tight_layout()
    plt.show()

def run_s2_experiments(agent_classes, labels):
    ps = np.arange(0.1, 0.91, 0.05)  # or whatever p-values you want
    regrets = {label: [] for label in labels}

    for p in ps:
        probs = [p, p + 0.1]
        for agent_class, label in zip(agent_classes, labels):
            agent, _ = run_agent(agent_class, probs)
            # Final regret after all steps
            regrets[label].append(np.max(probs) * agent.time_to_run - sum(agent.rewards))

    # Plotting
    plt.figure(figsize=(10, 6))
    for label in labels:
        plt.plot(ps, regrets[label], marker='o', label=label)
    plt.xlabel('p')
    plt.ylabel('Final Regret after T steps')
    plt.title('Final Regret vs p for S2 Experiments')
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    # --- S1: Single Bandit Game ---
    probs = [0.23, 0.55, 0.76, 0.44]
    optimal_arm = np.argmax(probs)

    agent_classes = [EpsilonGreedyAgent, KLUCBAgent, UCBAgent, ThompsonSamplingAgent]
    labels = ['Epsilon-Greedy', 'KL-UCB', 'UCB1', 'Thompson']

    agents = []
    for cls in agent_classes:
        if cls == EpsilonGreedyAgent:
            agent, _ = run_agent(cls, probs, epsilon=0.05)
        else:
            agent, _ = run_agent(cls, probs)
        agents.append(agent)

    plot_reward_regret_curves(agents, labels)
    plot_optimal_arm_counts(agents, labels, optimal_arm)

    # --- S2: Two-Armed Bandits with p and p+0.1 ---
    run_s2_experiments(agent_classes, labels)
