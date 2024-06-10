import pickle
import random
import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import arguments

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def test(args):
    if args.model_name == 'DQN':
        test_dqn(args)
    else:
        test_(args)

class DQN(nn.Module):
    def __init__(self, num_inputs, num_outputs, args):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(num_inputs, args.hidden_dim)
        self.fc2 = nn.Linear(args.hidden_dim, args.hidden_dim)
        self.fc3 = nn.Linear(args.hidden_dim, num_outputs)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

def one_hot_encode(state, state_dim):
    one_hot = np.zeros(state_dim)
    one_hot[state] = 1
    return one_hot

def test_(args, num_episodes=1):
    print("test start")
    if args.env_name == 'frozenlake':
        env = gym.make('FrozenLake-v1', desc=None, map_name="4x4", is_slippery=False)
    elif args.env_name == 'cliffwalking':
        env = gym.make('CliffWalking-v0')
    elif args.env_name == 'taxi':
        env = gym.make('Taxi-v3')#render_mode='human'

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
            next_state, reward, done, _, _ = env.step(action)
            episode_rewards += reward
            steps += 1
            state = next_state

            if done:
                if args.env_name == 'cliffwalking':
                    if reward != -100:  # CliffWalking environment success criteria
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

def test_dqn(args):
    print("dqn test start")
    if args.env_name == 'frozenlake':
        env = gym.make('FrozenLake-v1', desc=None, map_name="4x4", is_slippery=False, max_episode_steps=400)
    elif args.env_name == 'cliffwalking':
        env = gym.make('CliffWalking-v0', max_episode_steps=200)
    elif args.env_name == 'taxi':
        env = gym.make('Taxi-v3', max_episode_steps=400)

    state_dim = env.observation_space.n
    action_dim = env.action_space.n
    q = DQN(state_dim, action_dim, args).to(device)
    q.load_state_dict(torch.load(f"models/{args.model_name}_{args.env_name}_dqn_model.pt"))

    rewards_per_episode = []

    for episode in range(1):
        state = env.reset()[0]
        state = one_hot_encode(state, state_dim)
        done = False
        episode_rewards = 0

        while not done:
            action = epsilon_greedy_policy_dqn(state, q, 0, env)  # Use greedy policy for testing
            next_state, reward, done, _, _ = env.step(action)
            next_state = one_hot_encode(next_state, state_dim)
            state = next_state
            episode_rewards += reward

        rewards_per_episode.append(episode_rewards)

    print(f"Average reward over {args.num_episodes} episodes: {np.mean(rewards_per_episode)}")

def epsilon_greedy_policy_dqn(state, model, epsilon, env):
    if random.uniform(0, 1) < epsilon:
        return env.action_space.sample()
    else:
        with torch.no_grad():
            state = torch.tensor([state], device=device, dtype=torch.float32)
            q_values = model(state)
            return q_values.argmax().item()
