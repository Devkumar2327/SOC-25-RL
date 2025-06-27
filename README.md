# Task2-MultiArmedBandits

After week 1 and having a taken a look at the videos in week 2 you are ready to see some actual coding action.
The goal is to implement KL-UCB, UCB and Thompson Sampling agents and compare their performance. The correctness will be based on the final graph you get in main.py

### Definitive Goals
- Using `epsilon_greedy.py` as a reference, write code in `klucb.py`, `ucb.py` and `thompson.py` so that agents following that algorithm can play the game correctly.
- Each agent is derived from the baseclass in `base.py`. Please do not make changes to `epsilon_greedy.py` and `base.py`
- After you have implemented all the algorithms individually and tested them using the `__main__` parts of the respective files, plot the common curves of cumulative reward and regret for all the 4 agents by invoking them appropriately. This is described in detail below.
- As an exercise, what each method under the class does is left unexplained. I expect that you look at `epsilon_greedy.py` and try to understand what the general expectations from each function would be.
- Making a commit on the main branch counts as a submission

### Conventions
- A bandit is a single machine which when activated, throws a +1 with a chance $p$ and a 0 with chance $1-p$
- A bandit game (or just 'game' herein) is a scenario where you are offered, at each timestep upto the horizon $T$, the choice to draw from exactly one of the bandits. The chosen bandit dispenses a reward as previously described.

### What's required in `main.py` ?
Once all agents are ready, let them all try their hands at the same games (each run should result the same reward sequence for every agent playing the game). 

The first set of games, S1, will be vanilla (again 30k time steps). This set just contains the single bandit game given in the main section of any of those files you saw. Run that instance and plot a single curve (one for cumulative regret, one for cumulative reward) for all the 4 agents. Also plot a bar chart of the number of times each agent chose the optimal arm alone.


The next set of games we'll call S2 include a pair of bandits of probability $p$ and $p+0.1$ where $p \in \{0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9\}. I.e., in each game, there'll be two bandits, and there'll be 17 such games. For S2, cumulative reward and regret curves are not required to be plotted. What will be required is a plot of the final regret at the end of the games for each value of $p$. This should be easy. The hard part is to explain why the curves you obtained for KL-UCB and UCB look the way they look. Create a README file called `answers.md` and try to answer this riddle. All games last for 30k time steps. You might need hints for this, do ask if you need some!

Set S1 and S2 are required parts to be completed. This concludes the Week 2 Task description.

### Extra Miles
This is the zone of challenges. This part is not required but will tickle your brain.

Try to simulate a set S3, of games each with 2 bandits, the first bandit remains fixed at $p1 =0.9$, The second bandit takes probabilities between $0.1$ to $0.9$ (both inclusive) in steps of $0.05$. Plot the regret at the last time step for each $p2$. Can you explain the peculiar curve obtained for UCB and KL-UCB? This problem doesn't require Thompson Sampling. You can add code for this in `main.py` and also try answering this in `answers.md`

This last one is only a theoretical dilemma. Imagine that you're presented with a new type of bandit called 'the layer 2 bandit'. Now, when you chose a 'layer 2 bandit' to try your luck, what actually happens is, with 50% probability, it'll toss a fair coin and check for heads or tails. If heads, you get +1 otherwise, 0. With the other 50% probability it just triggers a regular bandit with win chance $p$. Suppose you have a bandit game with $n$ such layer 2 bandits. Will Thompson Sampling and KL-UCB still work well? How better or worse will they do? Explain theoretically in `answers.md`.
