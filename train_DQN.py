# Default libraries
import math
import random
import os
import time
import numpy as np
rng = np.random.default_rng()
import matplotlib.pyplot as plt
from collections import namedtuple, deque

# Gymnasium
import gymnasium as gym
print("gym version:", gym.__version__)

# pytorch
import torch
import torch.nn.functional as F

# Select hardware: 
if torch.cuda.is_available(): # GPU
    device = torch.device("cuda")
elif torch.backends.mps.is_available(): # Metal (Macos)
    device = torch.device("mps")
else: # CPU
    device = torch.device("cpu")
print(f"Device: {device}")

class MLP(torch.nn.Module):
    "Value network for DQN on Cartpole."

    def __init__(self, nb_observations, nb_hidden1, nb_hidden2, nb_actions):
        super(MLP, self).__init__()
        
        # Layers
        self.fc1 = torch.nn.Linear(nb_observations, nb_hidden1)
        self.fc2 = torch.nn.Linear(nb_hidden1, nb_hidden2)
        self.fc3 = torch.nn.Linear(nb_hidden2, nb_actions)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

# Named tuples are fancy dictionaries
Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))

# Replay buffer
class ReplayMemory(object):
    "Simple Experience Replay Memory using uniform sampling."

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def append(self, state, action, reward, next_state, done):
        "Appends a transition (s, a, r, s', done) to the buffer."

        # Get numpy arrays even if it is a torch tensor
        if isinstance(state, (torch.Tensor,)): state = state.numpy(force=True)
        if isinstance(next_state, (torch.Tensor,)): next_state = next_state.numpy(force=True)
        
        # Append to the buffer
        self.memory.append(Transition(state, action, reward, next_state, done))

    def sample(self, batch_size):
        "Returns a minibatch of (s, a, r, s', done)"

        # Sample the batch
        transitions = random.sample(self.memory, batch_size)
        
        # Transpose the batch.
        batch = Transition(*zip(*transitions))
        
        # Cast to tensors
        states = torch.tensor(batch.state, dtype=torch.float32, device=device)
        actions = torch.tensor(batch.action, dtype=torch.long, device=device)
        rewards = torch.tensor(batch.reward, dtype=torch.float32, device=device)
        next_states = torch.tensor(batch.next_state, dtype=torch.float32, device=device)
        dones = torch.tensor(batch.done, dtype=torch.bool, device=device)

        return Transition(states, actions, rewards, next_states, dones)

    def __len__(self):
        return len(self.memory)

# DQN algorithm
class DQNAgent:
    "DQN agent."
    
    def __init__(self, env, config):

        # Parameters
        self.env = env
        self.config = config

        # Number of actions
        self.n_actions = self.env.action_space.n

        # Number of states
        self.state, info = self.env.reset()
        self.n_observations = len(self.state)

        # Value network
        self.value_net = MLP(self.n_observations, config['nb_hidden'], config['nb_hidden'], self.n_actions).to(device)

        # Target network
        self.target_net = MLP(self.n_observations, config['nb_hidden'], config['nb_hidden'], self.n_actions).to(device)

        # Copy the value weights into the target network
        self.target_net.load_state_dict(self.value_net.state_dict())

        # Optimizer
        self.optimizer = torch.optim.Adam(self.value_net.parameters(), lr=self.config['learning_rate'])

        # Loss function
        self.loss_function = torch.nn.MSELoss()
        
        # Replay buffer
        self.memory = ReplayMemory(self.config['buffer_limit'])

        self.steps_done = 0
        self.episode_durations = []


    def act(self, state):

        # Decay epsilon exponentially
        self.epsilon = self.config['eps_end'] + (self.config['eps_start'] - self.config['eps_end']) * math.exp(-1. * self.steps_done / self.config['eps_decay'])

        # Keep track of time
        self.steps_done += 1
    
        # epsilon-greedy action selection
        if rng.random() < self.epsilon:
            return self.env.action_space.sample()
        else:
            with torch.no_grad():
                return self.value_net(state).argmax(dim=0).item()

    def update(self):

        # Only learn when the replay buffer is full enough
        if len(self.memory) < 2 * self.config['batch_size']:
            return
        
        # Sample a batch
        batch = self.memory.sample(self.config['batch_size'])

        # Compute Q(s_t, a) with the current value network.
        Q_values = self.value_net(batch.state)[range(self.config['batch_size']), batch.action]
        
        # Compute Q(s_{t+1}, a*) for all next states.
        # If the next state is terminal, set the value to zero.
        # Do not compute gradients.
        with torch.no_grad():
            next_Q_values = self.target_net(batch.next_state).max(dim=1).values
            next_Q_values[batch.done] = 0.0

        # Compute the target Q values
        targets = (next_Q_values * self.config['gamma']) + batch.reward

        # Compute loss
        loss = self.loss_function(Q_values, targets)

        # Reinitialize the gradients
        self.optimizer.zero_grad()

        # Backpropagation
        loss.backward()

        # In-place gradient clipping (optional)
        #torch.nn.utils.clip_grad_value_(self.value_net.parameters(), 100)

        # Optimizer step
        self.optimizer.step()

    def train(self, num_episodes):
        
        for i_episode in range(num_episodes):

            tstart = time.time()

            # Initialize the environment and get its state
            state, _ = self.env.reset()

            # Transform the state into a tensor
            state = torch.tensor(state, dtype=torch.float32, device=device)

            done = False
            steps_episode = 0
            while not done:
                
                # Select an action
                action = self.act(state)
                
                # Perform the action
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                
                # Terminal state
                done = terminated or truncated

                # Store the transition in memory
                self.memory.append(state, action, reward, next_state, done)

                # Move to the next state
                state = torch.tensor(next_state, dtype=torch.float32, device=device)

                # Perform one step of the optimization (on the policy network)
                self.update()

                # Update of the target network's weights
                if self.steps_done % self.config['target_update_period'] == 0:
                    self.target_net.load_state_dict(self.value_net.state_dict())

                # Finish episode
                steps_episode += 1
                if done:
                    self.episode_durations.append(steps_episode)
                    print(f"Episode {i_episode+1}, duration {steps_episode}, epsilon {self.epsilon:.4f} done in {time.time() - tstart}")


def main():
    # Hyperparameters
    config = {}
    config['nb_hidden'] = 128 # number of hidden neurons in each layer
    config['batch_size'] = 128 # number of transitions sampled from the replay buffer
    config['gamma'] = 0.99 # discount factor
    config['eps_start'] = 0.9 # starting value of epsilon
    config['eps_end'] = 0.05 # final value of epsilon
    config['eps_decay'] = 1000 # rate of exponential decay of epsilon, higher means a slower decay
    config['learning_rate'] = 1e-3 # learning rate of the optimizer
    config['target_update_period'] = 120 # update period (in steps) of the target network
    config['buffer_limit'] = 10000 # maximum number of transitions in the replay buffer

    # Create the environment
    env = gym.make('CartPole-v0')

    # Create the agent
    agent = DQNAgent(env, config)

    # Train the agent
    agent.train(num_episodes=250)

    plt.figure(figsize=(10, 6))
    plt.plot(agent.episode_durations)
    plt.xlabel("Episodes")
    plt.ylabel("Returns")
    plt.show()


if __name__ == "__main__":
    main()
