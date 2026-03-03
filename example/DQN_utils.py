import torch
import torch.optim as optim
import numpy as np
import random
import os
import pickle
import gymnasium as gym
from typing import List, Callable
from collections import deque
import itertools
from tqdm.notebook import tqdm

from student_client import create_student_gym_env, StudentGymEnv


# def format_state(state, target_length = 10):
#     """
#     Format observation to make it the same all the time. State can have shape (9,), (1, 9) or (10, 9).
#     This function pad the state and flatten it. Leads to vector of shape (90,)
#     """
#     state = np.array(state)
#     if len(state.shape) == 1: # shape (9,) -> (1, 9)
#         state = state.reshape(1, -1)

#     current_length = state.shape[0]
#     if current_length < target_length: # pad if less than 10
#         pad_size = target_length - current_length
#         # pad by repeating first vector
#         padding = np.repeat(state[0:1, :], pad_size, axis=0)
#         state = np.vstack((padding, state))

    # return state.flatten()

def format_state(state, end_time_step: int, target_length = 10):
    state = np.array(state)
    if len(state.shape) == 1: 
        state = state.reshape(1, -1)

    current_length = state.shape[0]
    
    # Create a sequential array for the actual timesteps
    # Example: If end_time_step is 20 and we have 10 rows, the times are 11, 12, ..., 20.
    start_time = end_time_step - current_length + 1
    time_seq = np.arange(start_time, end_time_step + 1, dtype=np.float32).reshape(-1, 1)

    # Pad both the state and the time sequence if necessary
    if current_length < target_length: 
        pad_size = target_length - current_length
        
        # Pad state by repeating the first row
        state_padding = np.repeat(state[0:1, :], pad_size, axis=0)
        state = np.vstack((state_padding, state))
        
        # Pad time by repeating the oldest timestep (freezing time for the padded rows)
        time_padding = np.repeat(time_seq[0:1, :], pad_size, axis=0)
        time_seq = np.vstack((time_padding, time_seq))

    # Normalize the time sequence so huge numbers don't overwhelm the neural network
    time_seq = time_seq / 100.0 
    
    # Append the time column to the state matrix: (10, 9) + (10, 1) -> (10, 10)
    state_with_time = np.hstack((state, time_seq))

    return state_with_time.flatten()

class QNetwork(torch.nn.Module):
    def __init__(self, n_observations: int, n_actions: int, nn_l1: int, nn_l2: int):
        """
        Initialize a new instance of QNetwork.

        Parameters
        ----------
        n_observations : int
            The size of the observation space.
        n_actions : int
            The size of the action space.
        nn_l1 : int
            The number of neurons on the first layer.
        nn_l2 : int
            The number of neurons on the second layer.
        """
        super(QNetwork, self).__init__()
        self.network = torch.nn.Sequential(
            torch.nn.Linear(n_observations, nn_l1), 
            torch.nn.ReLU(),
            torch.nn.Linear(nn_l1, nn_l2),
            torch.nn.ReLU(),
            torch.nn.Linear(nn_l2, n_actions)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)
    

class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
        
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states),
            np.array(actions),
            np.array(rewards, dtype=np.float32),
            np.array(next_states),
            np.array(dones, dtype=np.float32)
        )
    
    def __len__(self):
        return len(self.buffer)

class SplitReplayBuffer:
    def __init__(self, capacity=50000):
        self.normal_buffer = deque(maxlen=capacity)
        self.failure_buffer = deque(maxlen=capacity // 10)
        self.capacity = capacity
    
    def push(self, state, action, reward, next_state, done):
        is_failure = done and (action == 0 or action == 1)
        
        transition = (state, action, reward, next_state, done)
        if is_failure:
            self.failure_buffer.append(transition)
        else:
            self.normal_buffer.append(transition)
            
    def sample(self, batch_size, failure_ratio=0.2):
        num_failures = int(batch_size * failure_ratio)
        
        num_failures = min(num_failures, len(self.failure_buffer))
        num_normal = batch_size - num_failures
        
        # Sample from both buffers
        batch = []
        if num_failures > 0:
            batch.extend(random.sample(self.failure_buffer, num_failures))
        batch.extend(random.sample(self.normal_buffer, num_normal))
        
        # Shuffle the combined batch so the network doesn't memorize the order
        random.shuffle(batch)
        
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states),
            np.array(actions),
            np.array(rewards, dtype=np.float32),
            np.array(next_states),
            np.array(dones, dtype=np.float32)
        )
    
    def __len__(self):
        return len(self.normal_buffer) + len(self.failure_buffer)
    
    def save(self, filepath: str):
        data = {
            'normal': self.normal_buffer,
            'failure': self.failure_buffer
        }
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
        print(f"Buffer saved successfully to {filepath}")
        print(f"Contains {len(self.normal_buffer)} normal steps and {len(self.failure_buffer)} failures.")

    def load(self, filepath: str):
        
        if os.path.exists(filepath):
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
                
            self.normal_buffer = deque(data['normal'], maxlen=self.capacity)
            self.failure_buffer = deque(data['failure'], maxlen=self.capacity // 10)
            
            print(f"Buffer loaded successfully from {filepath}")
            print(f"Loaded {len(self.normal_buffer)} normal steps and {len(self.failure_buffer)} failures.")
        else:
            print(f"Error: File '{filepath}' not found.")

def train_dqn(
    num_episodes: int = 500,
    batch_size: int = 64,
    gamma: float = 0.99,
    learning_rate: float = 1e-3,
    epsilon_start: float = 1.0,
    epsilon_end: float = 0.05,
    epsilon_decay: float = 0.995,
    target_update_freq: int = 10,
    save_buffer_every = 10,
    reward_scale = 1000.0,
    device: str = "cpu",
    checkpoint_network = None,
    checkpoint_buffer = None,
    checkpoint_optimizer = None,
    train_only = False,
):
    n_observations = 100 # 90
    n_actions = 3

    checkpoint_dir = "./checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)
    best_reward = -float('inf')
    
    q_network = QNetwork(n_observations, n_actions, 128, 64).to(device)
    target_network = QNetwork(n_observations, n_actions, 128, 64).to(device)

    if checkpoint_network is not None:
        q_network.load_state_dict(checkpoint_network.state_dict())
    target_network.load_state_dict(q_network.state_dict())
    
    if checkpoint_optimizer is not None:
        optimizer = checkpoint_optimizer
    else:
        optimizer = optim.Adam(q_network.parameters(), lr=learning_rate)
    


    if checkpoint_buffer is not None:
        memory = checkpoint_buffer
    else:
        memory = SplitReplayBuffer(capacity=50000)

    memory.load("air_palaiseau_buffer.pkl")
    
    epsilon = epsilon_start
    episode_rewards = []
    
    for episode in tqdm(range(num_episodes), desc="Training DQN"):
        env = create_student_gym_env(user_token='14S544O#BEiKzhK')

        raw_state, info = env.reset()

        current_engine_time = info.get('step', 0)
        state = format_state(raw_state, current_engine_time)

        total_reward = 0
        done = False
        time_since_last_repair = 0
        step = 0
        while not done:
            step += 1

            log_text = ""
            if random.random() < epsilon:
                action = env.action_space.sample() # Explore
                log_text += f"action: {action}"
            else:
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
                with torch.no_grad():
                    q_values = q_network(state_tensor)    
                action = q_values.argmax(dim=1).item() # Exploit
                log_text += f"action: {action}"
                log_text += f", {q_values}"

            
            print(log_text)
                
            raw_next_state, reward, terminated, truncated, _ = env.step(action)

            current_engine_time = info.get('step', current_engine_time + 10)

            next_state = format_state(raw_next_state, current_engine_time)
            done = terminated or truncated

            if action == 1:
                if time_since_last_repair < 5:
                    reward = -5000.0
                time_since_last_repair = 0
            else:
                time_since_last_repair += 1

            scaled_reward = reward / reward_scale
   
            memory.push(state, action, scaled_reward, next_state, done)
            state = next_state
            total_reward += reward
            
            if len(memory) > batch_size:
                states_b, actions_b, rewards_b, next_states_b, dones_b = memory.sample(batch_size)
                
                states_tensor = torch.FloatTensor(states_b).to(device)
                actions_tensor = torch.LongTensor(actions_b).unsqueeze(1).to(device)
                rewards_tensor = torch.FloatTensor(rewards_b).unsqueeze(1).to(device)
                next_states_tensor = torch.FloatTensor(next_states_b).to(device)
                dones_tensor = torch.FloatTensor(dones_b).unsqueeze(1).to(device)
                
                current_q_values = q_network(states_tensor).gather(1, actions_tensor)
                
                with torch.no_grad():
                    best_next_actions = q_network(next_states_tensor).argmax(dim=1).unsqueeze(1)
                    target_q_values_next = target_network(next_states_tensor).gather(1, best_next_actions)
                    
                    target_q_values = rewards_tensor + (1 - dones_tensor) * gamma * target_q_values_next
                
                loss = torch.nn.MSELoss()(current_q_values, target_q_values)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
        epsilon = max(epsilon_end, epsilon * epsilon_decay)
        episode_rewards.append(total_reward)

        if total_reward > best_reward:
            best_reward = total_reward
            best_model_path = os.path.join(checkpoint_dir, "best_q_network.pth")
            torch.save(q_network.state_dict(), best_model_path)
            print(f"*** New best reward: {best_reward:.2f}! Model saved to {best_model_path} ***")

        
        if episode > 0 and episode % 50 == 0:
            periodic_model_path = os.path.join(checkpoint_dir, f"q_network_ep_{episode}.pth")
            torch.save({
                'episode': episode,
                'model_state_dict': q_network.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'reward': total_reward
            }, periodic_model_path)
        
        if episode % target_update_freq == 0:
            target_network.load_state_dict(q_network.state_dict())
            
        if episode % 1 == 0:
            print(f"Episode {episode} | Reward: {total_reward:.2f} | Epsilon: {epsilon:.3f} | Steps: {step}")

        if episode % save_buffer_every == 0:
            memory.save("air_palaiseau_buffer.pkl")
            
    return q_network, episode_rewards, memory



def train_only_dqn(
    num_epochs = 100,
    batch_size: int = 64,
    gamma: float = 0.99,
    learning_rate: float = 1e-3,
    target_update_freq: int = 10,
    device: str = "cpu",
    reward_scale = 1000.0,
    checkpoint_network = None,
    checkpoint_buffer = None,
):
    n_observations = 100
    n_actions = 3
    
    q_network = QNetwork(n_observations, n_actions, 128, 64).to(device)
    target_network = QNetwork(n_observations, n_actions, 128, 64).to(device)

    if checkpoint_network is not None:
        q_network.load_state_dict(checkpoint_network.state_dict())
    target_network.load_state_dict(q_network.state_dict())
    
    optimizer = optim.Adam(q_network.parameters(), lr=learning_rate)


    if checkpoint_buffer is not None:
        memory = checkpoint_buffer
    else:
        memory = SplitReplayBuffer(capacity=50000)

    memory.load("air_palaiseau_buffer.pkl")

    if len(memory) < batch_size:        
        print("Not enough memory")
        return
    
    # Training loop
    for i in range(num_epochs):
    
        states_b, actions_b, rewards_b, next_states_b, dones_b = memory.sample(batch_size)
        
        states_tensor = torch.FloatTensor(states_b).to(device)
        actions_tensor = torch.LongTensor(actions_b).unsqueeze(1).to(device)
        rewards_tensor = torch.FloatTensor(rewards_b).unsqueeze(1).to(device)
        next_states_tensor = torch.FloatTensor(next_states_b).to(device)
        dones_tensor = torch.FloatTensor(dones_b).unsqueeze(1).to(device)
        
        current_q_values = q_network(states_tensor).gather(1, actions_tensor)
        
        with torch.no_grad():
            best_next_actions = q_network(next_states_tensor).argmax(dim=1).unsqueeze(1)
            target_q_values_next = target_network(next_states_tensor).gather(1, best_next_actions)
            
            target_q_values = rewards_tensor / reward_scale + (1 - dones_tensor) * gamma * target_q_values_next
        
        loss = torch.nn.MSELoss()(current_q_values, target_q_values)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"Epoch {i+1}/{num_epochs} | Loss: {loss:.2f}")
                
        if i % target_update_freq == 0:
            target_network.load_state_dict(q_network.state_dict())
            
    return q_network