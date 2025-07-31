import argparse
import numpy as np
from collections import defaultdict

import itertools
import matplotlib.pyplot as plt
import os

# --- Constants & Helpers ---
GRID_ROWS, GRID_COLS = 4, 4
GOAL_SQUARES = [8, 12] 
NUM_ACTIONS = 10
MOVE_DIRECTIONS = {'L': (-1, 0), 'R': (1, 0), 'U': (0, -1), 'D': (0, 1)}
TERMINAL_STATES = ['GOAL', 'LOST']

# --- Helper Functions (square_to_coords, etc.) ---
def square_to_coords(square):
    y = (square - 1) // GRID_COLS
    x = (square - 1) % GRID_COLS
    return (x, y)

def coords_to_square(coords):
    x, y = coords
    if not (0 <= x < GRID_COLS and 0 <= y < GRID_ROWS): return None
    return y * GRID_COLS + x + 1

def is_between(p1_coords, p2_coords, r_coords):
    x1, y1 = p1_coords; x2, y2 = p2_coords; xr, yr = r_coords
    in_bbox = min(x1, x2) <= xr <= max(x1, x2) and min(y1, y2) <= yr <= max(y1, y2)
    if not in_bbox: return False
    return (yr - y1) * (x2 - x1) == (xr - x1) * (y2 - y1)

def get_args():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(description="MDP Planner for Half Field Offense.")
    parser.add_argument("--p", type=float, required=True, help="Skill parameter p for movement.")
    parser.add_argument("--q", type=float, required=True, help="Skill parameter q for passing/shooting.")
    parser.add_argument("--policy", type=str, required=True, help="Path to the opponent policy file (e.g., policy_greedy.txt).")
    return parser.parse_args()
# --- Policy Loading ---
def load_policy(filepath):
    """Loads an opponent policy from a text file."""
    policy = {}
    with open(filepath, 'r') as f:
        for line in f:
            parts = line.strip().split()
            state = tuple(map(int, parts[:4]))
            probs = list(map(float, parts[4:]))
            policy[state] = probs
    return policy

# --- Planner Class (from previous steps) ---
class Planner:
    def __init__(self, p, q, opponent_policy):
        self.p = p
        self.q = q
        self.opponent_policy = opponent_policy
        self.states = self._generate_states()
        # Initialize value function for all states, including terminal ones
        self.value_function = defaultdict(float)

    def _generate_states(self):
        states = []
        squares = range(1, 17)
        for b1, b2, r in itertools.permutations(squares, 3):
            states.append((b1, b2, r, 1))
            states.append((b1, b2, r, 2))
        return states

    def run_policy_iteration(self, gamma=1.0, theta=1e-5):
    
        print("Running Policy Iteration...")
        
        # 1. Start with a random policy (always take action 0)
        policy = {s: 0 for s in self.states}
        
        while True:
            # --- 2. Policy Evaluation Step ---
            print("  Evaluating policy...")
            # Iteratively solve for V_pi until it converges
            while True:
                delta = 0
                for s in self.states:
                    v = self.value_function[s]
                    # Get the action specified by the current policy
                    action = policy[s]
                    # Calculate the value based on that action's outcome
                    self.value_function[s] = self._calculate_q_value(s, action, gamma)
                    delta = max(delta, abs(v - self.value_function[s]))
                
                if delta < theta:
                    break # Value function for this policy has converged

            # --- 3. Policy Improvement Step ---
            print("  Improving policy...")
            policy_stable = True
            for s in self.states:
                old_action = policy[s]
                
                # Find the best action by looking one step ahead (greedy)
                action_values = {a: self._calculate_q_value(s, a, gamma) for a in range(NUM_ACTIONS)}
                best_action = max(action_values, key=action_values.get)
                
                # Update the policy if we found a better action
                if old_action != best_action:
                    policy[s] = best_action
                    policy_stable = False

            # If the policy did not change in a full sweep, it's optimal
            if policy_stable:
                print("Policy Iteration converged.")
                break
                
        # The final value function is already computed from the last evaluation
        # The optimal policy is also stored in the `policy` dictionary
        return policy, self.value_function

    def _calculate_q_value(self, state, action, gamma):
        q_sa = 0
        outcomes = self.get_transition_outcomes(state, action)
        for next_state, prob in outcomes.items():
            reward = 1.0 if next_state == 'GOAL' else 0.0
            q_sa += prob * (reward + gamma * self.value_function[next_state])
        return q_sa

    # (get_transition_outcomes and all _handle_* methods remain exactly the same)
    def get_transition_outcomes(self, state, action):
        outcomes = defaultdict(float)
        b1_sq, b2_sq, r_sq, ball_poss = state
        opponent_moves_probs = self.opponent_policy.get(state, [0.25, 0.25, 0.25, 0.25])
        for i, r_move_dir in enumerate(['L', 'R', 'U', 'D']):
            prob_r_move = opponent_moves_probs[i]
            if prob_r_move == 0: continue
            r_coords = square_to_coords(r_sq)
            r_move_delta = MOVE_DIRECTIONS[r_move_dir]
            r_next_coords = (r_coords[0] + r_move_delta[0], r_coords[1] + r_move_delta[1])
            r_next_sq = coords_to_square(r_next_coords)
            if r_next_sq is None: r_next_sq = r_sq
            action_outcomes = {}
            if 0 <= action <= 7: action_outcomes = self._handle_move(state, action, r_next_sq)
            elif action == 8: action_outcomes = self._handle_pass(state, r_next_sq)
            elif action == 9: action_outcomes = self._handle_shoot(state, r_next_sq)
            for next_state, prob in action_outcomes.items(): outcomes[next_state] += prob * prob_r_move
        return outcomes
    def _handle_move(self, state, action, r_next_sq):
        b1_sq, b2_sq, r_sq, ball_poss = state
        outcomes = defaultdict(float)
        is_b1_moving = (0 <= action <= 3)
        player_id = 1 if is_b1_moving else 2
        start_sq = b1_sq if is_b1_moving else b2_sq
        teammate_sq = b2_sq if is_b1_moving else b1_sq
        move_dir = list(MOVE_DIRECTIONS.keys())[action % 4]
        move_delta = MOVE_DIRECTIONS[move_dir]
        start_coords = square_to_coords(start_sq)
        next_coords = (start_coords[0] + move_delta[0], start_coords[1] + move_delta[1])
        next_sq = coords_to_square(next_coords)
        if next_sq is None: return {'LOST': 1.0}
        if next_sq == teammate_sq: return {(b1_sq, b2_sq, r_next_sq, ball_poss): 1.0}
        has_ball = (player_id == ball_poss)
        prob_move_fails = self.p * 2 if has_ball else self.p
        prob_move_succeeds = 1.0 - prob_move_fails
        if prob_move_fails > 0: outcomes['LOST'] = prob_move_fails
        next_b1_sq = next_sq if is_b1_moving else b1_sq
        next_b2_sq = next_sq if not is_b1_moving else b2_sq
        if has_ball:
            tackle = (next_sq == r_next_sq) or (next_sq == r_sq and r_next_sq == start_sq)
            if tackle:
                outcomes[(next_b1_sq, next_b2_sq, r_next_sq, ball_poss)] += prob_move_succeeds * 0.5
                outcomes['LOST'] += prob_move_succeeds * 0.5
            else:
                outcomes[(next_b1_sq, next_b2_sq, r_next_sq, ball_poss)] += prob_move_succeeds
        else:
            if next_sq == r_next_sq:
                outcomes[(b1_sq, b2_sq, r_next_sq, ball_poss)] += prob_move_succeeds
            else:
                outcomes[(next_b1_sq, next_b2_sq, r_next_sq, ball_poss)] += prob_move_succeeds
        return outcomes
    def _handle_pass(self, state, r_next_sq):
        b1_sq, b2_sq, r_sq, ball_poss = state
        passer_sq = b1_sq if ball_poss == 1 else b2_sq
        receiver_sq = b2_sq if ball_poss == 1 else b1_sq
        passer_coords, receiver_coords = square_to_coords(passer_sq), square_to_coords(receiver_sq)
        dist = max(abs(passer_coords[0] - receiver_coords[0]), abs(passer_coords[1] - receiver_coords[1]))
        prob_success = self.q - 0.1 * dist
        if is_between(passer_coords, receiver_coords, square_to_coords(r_next_sq)): prob_success /= 2.0
        prob_success = max(0, prob_success)
        return {(b1_sq, b2_sq, r_next_sq, 3 - ball_poss): prob_success, 'LOST': 1.0 - prob_success}
    def _handle_shoot(self, state, r_next_sq):
        b1_sq, b2_sq, r_sq, ball_poss = state
        shooter_sq = b1_sq if ball_poss == 1 else b2_sq
        shooter_x = square_to_coords(shooter_sq)[0]
        prob_goal = self.q - 0.2 * (3 - shooter_x)
        if r_next_sq in GOAL_SQUARES: prob_goal /= 2.0
        prob_goal = max(0, prob_goal)
        return {'GOAL': prob_goal, 'LOST': 1.0 - prob_goal}

# --- Main Execution & Output Generation ---

def generate_graphs(policy_file='policy_greedy.txt'):
    """Generates and saves the two required plots."""
    print("\n--- Generating Graphs ---")
    start_state = (5, 9, 8, 1)
    opponent_policy = load_policy(policy_file)

    # Graph 1: Vary p, fix q
    p_values = np.linspace(0, 0.5, 6)
    q_fixed = 0.7
    win_probs_p = []
    for p in p_values:
        planner = Planner(p, q_fixed, opponent_policy)
        planner.run_policy_iteration()
        win_probs_p.append(planner.value_function[start_state])
    
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(p_values, win_probs_p, marker='o')
    plt.title(f'Win Probability vs. p (q={q_fixed})')
    plt.xlabel('Parameter p')
    plt.ylabel('Probability of Scoring Goal')
    plt.grid(True)

    # Graph 2: Vary q, fix p
    q_values = np.linspace(0.6, 1.0, 5)
    p_fixed = 0.3
    win_probs_q = []
    for q in q_values:
        planner = Planner(p_fixed, q, opponent_policy)
        planner.run_policy_iteration()
        win_probs_q.append(planner.value_function[start_state])

    plt.subplot(1, 2, 2)
    plt.plot(q_values, win_probs_q, marker='o', color='r')
    plt.title(f'Win Probability vs. q (p={p_fixed})')
    plt.xlabel('Parameter q')
    plt.grid(True)
    
    plt.tight_layout()
   
    plt.show()

def compare_policies(p=0.3, q=0.8):
    """Calculates win probability against the three opponent types."""
    print("\n--- Comparing Expected Goals Against Policies ---")
    start_state = (5, 9, 8, 1)
    policy_files = ['policy_greedy.txt', 'policy_park_the_bus.txt', 'policy_random.txt']

    for policy_file in policy_files:
        if not os.path.exists(policy_file):
            print(f"Warning: {policy_file} not found. Skipping.")
            continue
        print(f"\nEvaluating against {policy_file}...")
        opponent_policy = load_policy(policy_file)
        planner = Planner(p, q, opponent_policy)
        planner.run_policy_iteration()
        win_prob = planner.value_function[start_state]
        print(f"-> Expected goals (Win Probability) from {start_state}: {win_prob:.4f}")

def create_dummy_policy_files():
    """Creates placeholder policy files for demonstration."""
    if not os.path.exists("policy_random.txt"):
        with open("policy_random.txt", "w") as f:
            # For a random policy, we don't need to specify every state,
            # as the planner defaults to random [0.25, 0.25, 0.25, 0.25].
            # This file can be empty or have a few example lines.
            f.write("5 9 8 1 0.25 0.25 0.25 0.25\n")
    if not os.path.exists("policy_greedy.txt"):
        with open("policy_greedy.txt", "w") as f:
            # Dummy greedy: always move towards B1 (player with ball) at state (5,9,8,1)
            # B1 is at sq 5 (0,1), R is at sq 8 (3,1). R should move Left.
            f.write("5 9 8 1 1.0 0.0 0.0 0.0\n")
    if not os.path.exists("policy_park_the_bus.txt"):
         with open("policy_park_the_bus.txt", "w") as f:
            # Dummy PTB: at sq 8, just move Up/Down.
            f.write("5 9 8 1 0.0 0.0 0.5 0.5\n")
    print("Dummy policy files created/verified.")

if __name__ == "__main__":
    # Create dummy files for demonstration so the script is runnable
    #create_dummy_policy_files()

    # 1. Get arguments from the command line
    args = get_args()

    # 2. Load the specified opponent policy
    try:
        print(f"Loading opponent policy from: {args.policy}")
        opponent_policy = load_policy(args.policy)
    except FileNotFoundError:
        print(f"Error: Policy file not found at {args.policy}")
        exit()

    # 3. Initialize and run the planner with the given parameters
    planner = Planner(p=args.p, q=args.q, opponent_policy=opponent_policy)
    
    # 4. Run Value Iteration to find the ideal value function
    planner.run_policy_iteration()

    # 5. Output the result for the specific start state
    start_state = (5, 9, 8, 1) # B1@5, B2@9, R@8, B1 has ball
    win_probability = planner.value_function.get(start_state, "State not found")
    generate_graphs(args.policy)

    print("\n" + "="*25)
    print("      MDP RESULTS")
    print("="*25)
    print(f"Parameters: p={args.p}, q={args.q}")
    print(f"Opponent Policy: {args.policy}")
    print(f"\nIdeal Value (Win Probability) from state {start_state}: {win_probability:.4f}")
    print("="*25)