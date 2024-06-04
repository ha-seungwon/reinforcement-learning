import gym
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
import pickle

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


def epsilon_greedy_policy(state, env, epsilon, q_values, is_dqn=False):
    if random.uniform(0, 1) < epsilon:
        return env.action_space.sample()  # Explore: select a random action
    else:
        if is_dqn:
            return np.argmax(q_values)  # Q-values are already computed for DQN
        else:
            return np.argmax(q_values[state])  # Q-values from Q-table


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
        epsilon = args.epsilon
        num_episodes = args.num_episodes

        N = np.zeros((env.observation_space.n, env.action_space.n))  # Visit count for each state-action pair

        for episode in range(num_episodes):
            episode_rewards = 0 # Rewards received in this episode
            history=[]

            state = env.reset()[0]
            done = False
            iterate=0
            while not done:
                action = epsilon_greedy_policy(state, env, epsilon, q_table)  # Epsilon-greedy action selection
                next_state, reward, done, _, _ = env.step(action)
                history.append((state,action,reward))
                state = next_state
                episode_rewards+=reward
                iterate+=1

            # Update Q-values
            G = 0
            for t in range(len(history)-1,-1,-1):  # 여기서 백월드, 포월드가 있는건가? range(len(episode_states) - 1, -1, -1): forward 그건 진짜 잘 안된다.
                state = history[t][0]
                action = history[t][1]
                reward = history[t][2]

                # gamma is discount factor
                G = gamma * G + reward
                N[state][action] += 1
                alpha_t = 1 / N[state][action]  # Decaying alpha
                q_table[state][action] += alpha_t * (G - q_table[state][action])

            rewards_per_episode.append(episode_rewards)
            #print(f"episode {episode} done , episode reward : {episode_rewards}, episode mean reward {episode_rewards/iterate}, episode iteration {iterate}")

    elif args.model_name == 'TD':
        alpha = args.alpha  # Learning rate
        gamma = args.gamma  # Discount factor
        epsilon = args.epsilon  # Exploration rate
        num_episodes = args.num_episodes



        for episode in range(num_episodes):
            state = env.reset()[0]
            done = False
            episode_rewards = 0
            iterate = 0
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

                iterate+=1


            rewards_per_episode.append(episode_rewards)

            print(f"episode {episode} done , episode reward : {episode_rewards}, episode mean reward {episode_rewards/iterate}, episode iteration {iterate}")


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