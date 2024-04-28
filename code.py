# Import necessary libraries
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt  # Import matplotlib for plotting
import pandas as pd  # Import pandas for data manipulation

class PublicGoodsGame:
    def __init__(self, num_players, num_treatments):
        self.num_players = num_players
        self.num_treatments = num_treatments
        self.current_state = None
        self.total_contribution = 0
        self.round = 0

    def reset(self):
        # Reset the game state to the initial state and return it
        self.current_state = np.zeros(self.num_players)
        self.total_contribution = 0
        self.round = 0
        return self.current_state

    def step(self, action):
        # Apply the action (player's contribution) and update the state
        self.total_contribution += action
        reward = self._calculate_reward(action)
        self.current_state = self._get_next_state()
        self.round += 1
        done = self._check_if_done()
        return self.current_state, reward, done, {}

    def _get_next_state(self):
        # Update the state based on the total contribution
        return np.random.rand(self.num_players)

    def _calculate_reward(self, action):
        # Calculate the reward based on the action and the total contribution
        return action * 0.1

    def _check_if_done(self):
        # Check if the game is over (e.g., after a certain number of rounds)
        return self.round >= self.num_treatments

# Assuming a custom environment for the game
# Parameters
num_episodes = 10  # Total number of rounds
num_players = 6  # Number of players in the game grouped in 24 groups
num_treatments = 3  # Number of treatments
training_rounds = 8  # Number of rounds used for training
state_size = 6  # Define the size of the state based on the 6 players
action_size = 5  # Define the size of the action space based on the contribution choices
learning_rate_actor = 0.001
learning_rate_critic = 0.005

def build_actor_network():
    model = Sequential([
        Dense(128, activation='relu', input_dim=state_size),
        Dense(action_size, activation='softmax')  # Use softmax for probability distribution
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate_actor),
                  loss='categorical_crossentropy')
    return model

def build_critic_network():
    model = Sequential([
        Dense(128, activation='relu', input_dim=state_size),
        Dense(1)  # Output a single value for the state-value function
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate_critic),
                  loss='mean_squared_error')
    return model

# Initialize the Public Goods Game environment with the specified number of players and treatments
environment = PublicGoodsGame(num_players=num_players, num_treatments=num_treatments)

# Initialize actor and critic networks
actor_model = build_actor_network()
critic_model = build_critic_network()

# Training loop
for episode in range(num_episodes):
    current_state = environment.reset()
    total_reward = 0

for round in range(training_rounds):
        # Actor network decides on action based on current state
        action_probabilities = actor_model.predict(current_state.reshape(1, -1))
        action = np.random.choice(np.arange(len(action_probabilities[0])), p=action_probabilities[0])

     # Environment executes action and returns next state and reward
        next_state, reward, done, _ = environment.step(action)
        total_reward += reward

        if done:
            break

        # Critic network updates
        target = reward + critic_model.predict(next_state.reshape(1, -1))
        predicted = critic_model.predict(current_state.reshape(1, -1))
        critic_loss = tf.reduce_mean(tf.square(target - predicted))

        current_state = next_state
        max_steps_per_episode = 100

# Define state_samples, action_samples, and rewards_samples
state_samples = []
action_samples = []
rewards_samples = []

# Loop over episodes and steps within each episode to collect data
for episode in range(num_episodes):
    current_state = environment.reset()
    episode_states = []
    episode_actions = []
    episode_rewards = []

    for step in range(max_steps_per_episode):

 # Store the state, action, and reward
        episode_states.append(current_state)
        episode_actions.append(action)
        episode_rewards.append(reward)

# After the episode ends, add the episode's data to the samples
    state_samples.extend(episode_states)
    action_samples.extend(episode_actions)
    rewards_samples.extend(episode_rewards)

state_samples = np.array(state_samples)
action_samples = np.array(action_samples)
rewards_samples = np.array(rewards_samples)
if action_samples.ndim == 1:
    action_samples = action_samples.reshape(-1, 1)
combined_samples = np.concatenate([state_samples, action_samples], axis=1)
correct_state_samples = state_samples[:, :5]
correct_combined_samples = np.concatenate([correct_state_samples, action_samples], axis=1)
print("Shape of correct_combined_samples:", correct_combined_samples.shape)

# Train the critic model
history = critic_model.fit(correct_combined_samples, rewards_samples, epochs=100, verbose=0)

# Read the CSV file
pgg_data = pd.read_csv('OrderedColumnist_Imputed_SoLI_rightClmns.csv')

# Calculate average contribution per round
avg_contribution_per_round = pgg_data.groupby('Round')['Contribution'].mean()

# Generate a new model that follows a similar trend
rounds = np.arange(1, 11)
initial_contribution = avg_contribution_per_round.iloc[0]
decrease_rate = (avg_contribution_per_round.iloc[0] - avg_contribution_per_round.iloc[-1]) / 9  

# Linear decrease over rounds
mocked_contributions_adjusted = initial_contribution - decrease_rate * (rounds - 1)

# Plotting MSE over epochs
mse_history = history.history['loss']
plt.figure(figsize=(10, 6))
plt.plot(mse_history, label='MSE')
plt.title('MSE Over Training Epochs')
plt.xlabel('Epoch')
plt.ylabel('Mean Squared Error')
plt.legend()
plt.grid(True)
plt.savefig('mse_over_epochs.png')
plt.show()

# Output the final MSE
mse_final = mse_history[-1]
print(f'Final MSE: {mse_final}')
