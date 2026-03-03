import torch
import numpy as np
import gymnasium as gym
from tqdm.notebook import tqdm

from typing import Tuple, List

from student_client import create_student_gym_env



class PolicyNetwork(torch.nn.Module):
    """
    A neural network used as a policy for the REINFORCE algorithm.

    Attributes
    ----------
    layer1 : torch.nn.Linear
        A fully connected layer.

    Methods
    -------
    forward(state: torch.Tensor) -> torch.Tensor
        Define the forward pass of the PolicyNetwork.
    """

    def __init__(self, n_observations: int, n_actions: int):
        """
        Initialize a new instance of PolicyNetwork.

        Parameters
        ----------
        n_observations : int
            The size of the observation space.
        n_actions : int
            The size of the action space.
        """
        super(PolicyNetwork, self).__init__()

        # self.layer1 = torch.nn.Linear(n_observations, n_actions, bias=False)
        self.layer1 = torch.nn.Linear(n_observations, n_actions)
        # self.layer2 = torch.nn.Linear(64, n_actions)


    def forward(self, state_tensor: torch.Tensor) -> torch.Tensor:
        """
        Calculate the probability of each action for the given state.

        Parameters
        ----------
        state_tensor : torch.Tensor
            The input tensor (state).
            The shape of the tensor should be (N, dim),
            where N is the number of states vectors in the batch
            and dim is the dimension of state vectors.

        Returns
        -------
        torch.Tensor
            The output tensor (the probability of each action for the given state).
        """
        # relu = torch.nn.ReLU()
        x = self.layer1(state_tensor)
        # x = relu(x)
        # x = self.layer2(x)
        # x = relu(x)
        # print(f"Logits: {logits}")
        out = torch.nn.functional.softmax(x, dim=-1)
        # print(f"Softmax: {out}")
        return out
    


def sample_discrete_action(policy_nn: PolicyNetwork, state: np.ndarray, device) -> Tuple[int, torch.Tensor]:
    """
    Sample a discrete action based on the given state and policy network.

    This function takes a state and a policy network, and returns a sampled action and its log probability.
    The action is sampled from a categorical distribution defined by the output of the policy network.

    Parameters
    ----------
    policy_nn : PolicyNetwork
        The policy network that defines the probability distribution of the actions.
    state : np.ndarray
        The state based on which an action needs to be sampled.

    Returns
    -------
    Tuple[int, torch.Tensor]
        The sampled action and its log probability.
    """


    state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
    actions_probability_distribution_params = policy_nn(state_tensor)
    actions_probability_distribution = torch.distributions.Categorical(probs=actions_probability_distribution_params)
    sampled_action_tensor = actions_probability_distribution.sample()
    sampled_action = sampled_action_tensor.item()
    sampled_action_log_probability = actions_probability_distribution.log_prob(sampled_action_tensor)

    # print(f"State tensor: {state_tensor}")
    # print(f"    probas: {actions_probability_distribution_params}")
    # print(f"    action: {sampled_action}")
    # print(policy_nn.layer1.weight)

    return sampled_action, sampled_action_log_probability

def sample_one_episode(
    policy_nn: PolicyNetwork, max_episode_duration: int, device,
) -> Tuple[List[np.ndarray], List[int], List[float], List[torch.Tensor]]:
    """Execute one episode in `env` using the policy defined by `policy_nn`.

    Parameters
    ----------
    env : gym.Env
        The environment to play in.
    policy_nn : PolicyNetwork
        The policy neural network.
    max_episode_duration : int
        The maximum duration of the episode.

    Returns
    -------
    Tuple[List[np.ndarray], List[int], List[float], List[torch.Tensor]]
        The states, actions, rewards, and log-probability of the action for each time step in the episode.
    """
    env = create_student_gym_env(user_token='14S544O#BEiKzhK')
    state_t, info = env.reset()

    episode_states = []
    episode_actions = []
    episode_log_prob_actions = []
    episode_rewards = []
    episode_states.append(state_t)

    for t in range(max_episode_duration):
        action_t, log_prob_action_t = sample_discrete_action(policy_nn, state_t, device)
        state_t, reward_t, terminated, truncated, info = env.step(action_t)
        done = terminated or truncated

        # print(f"Log probs: {log_prob_action_t}")
        # print(f"    action: {action_t}")

        if len(state_t.shape) > 1:
            state_t = state_t[-1]

        # print(f"State: {state_t}")

        episode_states.append(state_t)
        episode_actions.append(action_t)
        episode_log_prob_actions.append(log_prob_action_t)
        episode_rewards.append(float(reward_t))

        if done:
            break

    return episode_states, episode_actions, episode_rewards, episode_log_prob_actions


def avg_return_on_multiple_episodes(
    policy_nn: PolicyNetwork,
    num_test_episode: int,
    max_episode_duration: int,
    device,
) -> float:
    """
    Play multiple episodes of the environment and calculate the average return.

    Parameters
    ----------
    policy_nn : PolicyNetwork
        The policy neural network.
    num_test_episode : int
        The number of episodes to play.
    max_episode_duration : int
        The maximum duration of an episode.

    Returns
    -------
    float
        The average return.
    """

    total_return = 0.0
    for episode_index in range(num_test_episode):
        episode_states, episode_actions, episode_rewards, episode_log_prob_actions = sample_one_episode(policy_nn, max_episode_duration, device) # TODO
        total_return += np.sum(episode_rewards)
    average_return = total_return / num_test_episode

    return average_return

def train_reinforce_discrete(
    num_train_episodes: int,
    num_test_per_episode: int,
    max_episode_duration: int,
    learning_rate: float,
    device,
) -> Tuple[PolicyNetwork, List[float]]:
    """
    Train a policy using the REINFORCE algorithm.

    Parameters
    ----------
    num_train_episodes : int
        The number of training episodes.
    num_test_per_episode : int
        The number of tests to perform per episode.
    max_episode_duration : int
        The maximum length of an episode.
    learning_rate : float
        The learning rate for the Adam optimizer.

    Returns
    -------
    Tuple[PolicyNetwork, List[float]]
        The final trained policy and the average evaluation returns computed after each training episode.
    """
    episode_avg_return_list = []

    policy_nn = PolicyNetwork(9, 3).to(device)
    optimizer = torch.optim.Adam(policy_nn.parameters(), lr=learning_rate)

    for episode_index in tqdm(range(num_train_episodes)):

        print("Sampling one full episode")
        _, _, episode_reward_list, episode_log_prob_action_list = sample_one_episode(policy_nn, max_episode_duration, device)
        print(f"Start training for episode {episode_index+1} / {num_train_episodes}")
        # for t in range(len(episode_reward_list)):
        #     future_return = np.sum(episode_reward_list[t:]) 
        #     returns_tensor = torch.tensor(future_return, dtype=torch.float32)

        #     log_prob_actions_tensor = episode_log_prob_action_list[t]
        #     # print("proba: ", log_prob_actions_tensor)

        #     loss = -returns_tensor * log_prob_actions_tensor

        #     optimizer.zero_grad()
        #     loss.backward()
        #     optimizer.step()

        #     # print("loss: ",loss.item())

        returns = []
        for t in range(len(episode_reward_list)):
            returns.append(np.sum(episode_reward_list[t:]))

        returns_tensor = torch.tensor(returns, dtype=torch.float32).to(device)
        returns_tensor = (returns_tensor - returns_tensor.mean()) / (returns_tensor.std() + 1e-8)

        policy_loss = []
        for t in range(len(episode_log_prob_action_list)):
            policy_loss.append(-returns_tensor[t] * episode_log_prob_action_list[t])
            
        loss = torch.stack(policy_loss).sum()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"loss: {loss.item()}")

        print("Training done")

        # Test the current policy
        print("Testing current pollicy")
        test_avg_return = avg_return_on_multiple_episodes(
            policy_nn=policy_nn,
            num_test_episode=num_test_per_episode,
            max_episode_duration=max_episode_duration,
            device=device
        )

        # Monitoring
        episode_avg_return_list.append(test_avg_return)

    return policy_nn, episode_avg_return_list