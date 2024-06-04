import gym
import numpy as np
import random
import argparse
import matplotlib.pyplot as plt
import warnings
import pickle
import torch
import torch.nn as nn
import torch.optim as optim

warnings.filterwarnings("ignore")


def epsilon_greedy_policy(state, env, epsilon, q_values, is_dqn=False):
    if random.uniform(0, 1) < epsilon:
        return env.action_space.sample()  # Explore: select a random action
    else:
        if is_dqn:
            return np.argmax(q_values)  # Q-values are already computed for DQN
        else:
            return np.argmax(q_values[state])  # Q-values from Q-table

class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x
def one_hot_encode(state, num_states):
    state_vector = np.zeros(num_states)
    state_vector[state] = 1
    return state_vector

def train(args):
    print("train start")
    if args.env_name == 'frozenlake':
        env = gym.make('FrozenLake-v1', desc=None, map_name="4x4", is_slippery=False)
    elif args.env_name == 'cliffwalking':
        env = gym.make('CliffWalking-v0')
    elif args.env_name == 'taxi':
        env = gym.make('Taxi-v3')

    env.reset()
    env.action_space.seed(42)

    # Initialize Q-table with zeros
    q_table = np.zeros((env.observation_space.n, env.action_space.n))
    rewards_per_episode = []

    if args.model_name == 'q_learning':
        alpha = args.alpha  # Learning rate
        gamma = args.gamma  # Discount factor
        epsilon = args.epsilon  # Exploration rate

        num_episodes = args.num_episodes
        for episode in range(num_episodes):
            state = env.reset()[0]
            done = False
            episode_rewards = 0

            while not done:
                action = epsilon_greedy_policy(state, env, epsilon, q_table)
                next_state, reward, done, _, _ = env.step(action)

                # Update Q-value using the Bellman equation
                old_value = q_table[state, action]
                next_max = np.max(q_table[next_state])
                new_value = old_value + alpha * (reward + gamma * next_max - old_value)
                q_table[state, action] = new_value

                state = next_state
                episode_rewards += reward

            rewards_per_episode.append(episode_rewards)

    elif args.model_name == 'MC':
        gamma = args.gamma
        alpha = args.alpha
        epsilon = args.epsilon
        num_episodes = args.num_episodes

        N = np.zeros((env.observation_space.n, env.action_space.n))  # Visit count for each state-action pair

        for episode in range(num_episodes):
            episode_states = []  # States visited in this episode
            episode_actions = []  # Actions taken in this episode
            episode_rewards = []  # Rewards received in this episode

            state = env.reset()[0]
            done = False

            while not done:
                action = epsilon_greedy_policy(state, env, epsilon, q_table)  # Epsilon-greedy action selection
                next_state, reward, done, _, _ = env.step(action)

                episode_states.append(state)
                episode_actions.append(action)
                episode_rewards.append(reward)

                state = next_state

            total_reward = sum(episode_rewards)
            rewards_per_episode.append(total_reward)

            # Update Q-values
            G = 0
            for t in range(len(episode_states) - 1, -1, -1):  # Loop through the episode in reverse
                state = episode_states[t]
                action = episode_actions[t]
                reward = episode_rewards[t]

                G = gamma * G + reward
                N[state][action] += 1
                alpha_t = 1 / N[state][action]  # Decaying alpha
                q_table[state][action] += alpha_t * (G - q_table[state][action])

    elif args.model_name == 'TD':
        alpha = args.alpha  # Learning rate
        gamma = args.gamma  # Discount factor
        epsilon = args.epsilon  # Exploration rate
        num_episodes = args.num_episodes

        for episode in range(num_episodes):
            state = env.reset()[0]
            done = False
            episode_rewards = 0

            while not done:
                action = epsilon_greedy_policy(state, env, epsilon, q_table)  # Epsilon-greedy action selection
                next_state, reward, done, _, _ = env.step(action)

                # TD(0) update
                old_value = q_table[state, action]
                next_max = np.max(q_table[next_state])
                new_value = old_value + alpha * (reward + gamma * next_max - old_value)
                q_table[state, action] = new_value

                state = next_state
                episode_rewards += reward

            rewards_per_episode.append(episode_rewards)

    elif args.model_name == 'sarsa':
        alpha = args.alpha  # Learning rate
        gamma = args.gamma  # Discount factor
        epsilon = args.epsilon  # Exploration rate

        num_episodes = args.num_episodes
        for episode in range(num_episodes):
            state = env.reset()[0]
            action = epsilon_greedy_policy(state, env, epsilon, q_table)
            done = False
            episode_rewards = 0

            while not done:
                next_state, reward, done, _, _ = env.step(action)
                next_action = epsilon_greedy_policy(next_state, env, epsilon, q_table)

                # Update Q-value using the SARSA update rule
                old_value = q_table[state, action]
                next_value = q_table[next_state, next_action]
                new_value = old_value + alpha * (reward + gamma * next_value - old_value)
                q_table[state, action] = new_value

                state, action = next_state, next_action
                episode_rewards += reward

            rewards_per_episode.append(episode_rewards)
    elif args.model_name == 'dqn':
        input_dim = env.observation_space.n
        output_dim = env.action_space.n

        dqn = DQN(input_dim, output_dim)
        optimizer = optim.Adam(dqn.parameters(), lr=args.alpha)
        loss_fn = nn.MSELoss()

        gamma = args.gamma
        epsilon = args.epsilon
        num_episodes = args.num_episodes

        for episode in range(num_episodes):
            print(f"episode {episode}")
            state = env.reset()[0]
            original_state = state
            state = torch.tensor(one_hot_encode(state, input_dim), dtype=torch.float32)
            done = False
            episode_rewards = 0

            while not done:
                q_values = dqn(state)
                action = epsilon_greedy_policy(state, env, epsilon, q_values.detach().cpu().numpy(), is_dqn=True)
                next_state, reward, done, _, _ = env.step(action)
                next_state = torch.tensor(one_hot_encode(next_state, input_dim), dtype=torch.float32)

                target = q_values.clone()
                if done:
                    target[action] = reward
                else:
                    next_q_values = dqn(next_state)
                    target[action] = reward + gamma * torch.max(next_q_values).item()

                loss = loss_fn(q_values, target)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                state = next_state
                episode_rewards += reward

            rewards_per_episode.append(episode_rewards)

    with open(f"models/{args.model_name}_{args.env_name}_q_table.pkl", "wb") as f:
        pickle.dump(q_table, f)

    return q_table, rewards_per_episode


def test(args, num_episodes=1):
    print("test start")
    if args.env_name == 'frozenlake':
        env = gym.make('FrozenLake-v1', render_mode='human', desc=None, map_name="4x4", is_slippery=False)
    elif args.env_name == 'cliffwalking':
        env = gym.make('CliffWalking-v0', render_mode='human')
    elif args.env_name == 'taxi':
        env = gym.make('Taxi-v3', render_mode='human')

    with open(f"models/{args.model_name}_{args.env_name}_q_table.pkl", "rb") as f:
        q_table = pickle.load(f)

    total_rewards = 0
    success_count = 0
    total_steps = 0

    for episode in range(num_episodes):
        state = env.reset()[0]
        done = False
        episode_rewards = 0
        steps = 0

        while not done:
            action = np.argmax(q_table[state])
            state, reward, done, _, _ = env.step(action)

            episode_rewards += reward
            steps += 1

            if done:
                print("done!!",reward)
                if args.env_name =='cliffwalking':
                    success_count += 1
                else:
                    if reward > 0:
                        success_count += 1

        total_rewards += episode_rewards
        total_steps += steps

    avg_rewards = total_rewards / num_episodes
    success_rate = success_count / num_episodes
    avg_steps = total_steps / num_episodes

    print(f"Average Rewards: {avg_rewards}")
    print(f"Success Rate: {success_rate * 100}%")
    print(f"Average Steps to Goal: {avg_steps}")



def plot_rewards(env_list, model_list, all_rewards):
    fig, axes = plt.subplots(len(env_list), len(model_list), figsize=(15, 15))
    fig.suptitle('Rewards per Episode for Different Environments and Models', fontsize=16)

    for i, env in enumerate(env_list):
        for j, model in enumerate(model_list):
            rewards = all_rewards[i][j]
            axes[i, j].plot(rewards)
            axes[i, j].set_title(f'{env} - {model}')
            axes[i, j].set_xlabel('Episodes')
            axes[i, j].set_ylabel('Rewards')

    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='q_learning', help='Name of the model')
    parser.add_argument('--env_name', type=str, default='cliffwalking', help='Name of the env')
    parser.add_argument('--alpha', type=float, default=0.1, help='Learning rate')
    parser.add_argument('--gamma', type=float, default=0.99, help='Discount factor')
    parser.add_argument('--epsilon', type=float, default=0.3, help='Exploration rate')
    parser.add_argument('--num_episodes', type=int, default=30000, help='Number of training episodes')
    args, unknown = parser.parse_known_args()

    env_list = ['cliffwalking', 'frozenlake', 'taxi']
    model_list = ["TD","q_learning", "sarsa" ]  # "TD" "MC" "q_learning", "sarsa" , "dqn"

    all_rewards = []

    for env in env_list:
        print(f"env is {env}")
        model_rewards_per_episode = []
        for model in model_list:
            print(f"model is {model}")
            args.env_name = env
            args.model_name = model
            _, rewards_per_episode = train(args)
            test(args)
            model_rewards_per_episode.append(rewards_per_episode)
        all_rewards.append(model_rewards_per_episode)

    plot_rewards(env_list, model_list, all_rewards)


if __name__ == '__main__':
    main()
