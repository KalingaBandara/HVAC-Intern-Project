import numpy as np
import random

class HVACController:
    def __init__(self, state_space, action_space, alpha=0.1, gamma=0.9, epsilon=0.1):
        self.state_space = state_space
        self.action_space = action_space
        self.alpha = alpha  # Learning rate
        self.gamma = gamma  # Discount factor
        self.epsilon = epsilon  # Exploration-exploitation trade-off

        # Q-table initialization
        self.q_table = {}
        for state in state_space:
            self.q_table[state] = {}
            for action in action_space:
                self.q_table[state][action] = 0

    def select_action(self, state):
        # Epsilon-greedy policy for action selection
        if random.uniform(0, 1) < self.epsilon:
            return random.choice(self.action_space)
        else:
            return max(self.q_table[state], key=self.q_table[state].get)

    def update_q_table(self, state, action, reward, next_state):
        # Q-learning update rule
        max_next_q_value = max(self.q_table[next_state].values())
        self.q_table[state][action] += self.alpha * (reward + self.gamma * max_next_q_value - self.q_table[state][action])

def calculate_cooling_load(T_chwr, T_chws, F_chw, C_p=4200, rho=1000):
    return (C_p * rho * F_chw * (T_chwr - T_chws)) / 3600

def calculate_reward(T_chwr, T_chwr_ref, x_comfort, x_energy,  P_chiller, P_chiller_nominal, beta1=0.25, beta2=4.15, beta3=-1, beta4=1):
    U_comfort = 1 / (1 + beta1 * np.exp(beta2 * (T_chwr - T_chwr_ref)))
    U_energy =  beta3 * ((np.sum(P_chiller))/(np.sum(P_chiller_nominal))) + beta4
    return U_comfort * x_comfort + U_energy * x_energy

def simulate_environment(state, action):
    cooling_load, T_outdoor, chiller_status = state
    T_chws_ref = action

    P_chiller = (cooling_load + a1*T_chws + a2*(1 - T_chws/ T_cws))*(T_cws/T_chws - a3*cooling_load) - cooling_load
    

    return 


# Example usage
state_space = [(cooling_load, T_outdoor, chiller_status) for cooling_load in range(10, 100, 10)
                                                                for T_outdoor in range(20, 30, 2)
                                                                for chiller_status in [0, 1]]

beta5 = -0.25
beta6 = 20
T_chws_ref = beta5 * T_outdoor + beta6
action_space = [T_chws_ref + i for i in range(-1, 2)]  # Assuming T_ref is defined

controller = HVACController(state_space, action_space)

initial_cooling_load = 1000
initial_outdoor_temp = 24
initial_chiller_status = 1
max_timesteps = 10

# Training loop
num_episodes = 1000
for episode in range(num_episodes):
    # Reset the environment to the initial state
    state = (initial_cooling_load, initial_outdoor_temp, initial_chiller_status)
    total_reward = 0

    for _ in range(max_timesteps):
        # Select an action
        action = controller.select_action(state)

        # Simulate the environment and get the next state and reward
        next_state = simulate_environment(state, action)
        reward = calculate_reward(T_chwr, T3, x1, x2)

        # Update the Q-table
        controller.update_q_table(state, action, reward, next_state)

        state = next_state
        total_reward += reward

    # ... (log and save results, etc.)
