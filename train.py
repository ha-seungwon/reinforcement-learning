import gym
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
import pickle
import collections
import torch.nn.functional as F
import arguments

buffer_limit = 10000


def train(args):
    print("train start")
    if args.env_name == 'frozenlake':
        env = gym.make('FrozenLake-v1', desc=None, map_name="4x4", is_slippery=False)
    elif args.env_name == 'cliffwalking':
        env = gym.make('CliffWalking-v0')
    elif args.env_name == 'taxi':
        env = gym.make('Taxi-v3')#, render_mode='human') #, render_mode='human'

    # set arguments for each algorithm hyperparameter tuning

    print(args)

    env.reset()
    env.action_space.seed(42)

    # Initialize Q-table with zeros
    q_table = np.zeros((env.observation_space.n, env.action_space.n))
    rewards_per_episode = []

    if args.model_name == 'MC':
        N = np.zeros((env.observation_space.n, env.action_space.n))  # Visit count for each state-action pair
        k = 0
        epsilon = 1 if args.glie else args.epsilon

        print("epsilon",epsilon)

        for episode in range(args.num_episodes):
            episode_rewards = 0  # Rewards received in this episode
            history = []

            state = env.reset()[0]
            done = False
            iterate = 0

            while not done:
                action = epsilon_greedy_policy(state, env, epsilon, q_table)  # Epsilon-greedy action selection
                next_state, reward, done, _, _ = env.step(action)

                if args.env_name == 'frozenlake':
                    if done and reward != 1:
                        reward = -10
                    else:
                        reward = -1

                history.append((state, action, reward))
                state = next_state
                episode_rewards += reward
                iterate += 1

            # Update Q-values
            G = 0
            for t in range(len(history) - 1, -1, -1):
                state = history[t][0]
                action = history[t][1]
                reward = history[t][2]

                # gamma is discount factor
                G = args.gamma * G + reward
                N[state][action] += 1
                #alpha_t = 1 / N[state][action]  # Decaying alpha
                alpha_t = args.alpha  # Decaying alpha
                q_table[state][action] += alpha_t * (G - q_table[state][action])

            if args.glie and episode % args.glie_update_time == 0:
                k += 1
                epsilon = 1 / k

            rewards_per_episode.append(episode_rewards)
            if episode % 1000 == 0:
                print(f"Episode {episode} done, episode reward: {episode_rewards}, "
                      f"mean reward per iteration: {episode_rewards / iterate}, iterations: {iterate}, epsilon: {epsilon}")

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
                if args.env_name == 'frozenlake':
                    if done and reward != 1:
                        reward = -10
                    else:
                        reward = -1

                # TD(0) update
                old_value = q_table[state, action]
                next_max = np.max(q_table[next_state])
                new_value = old_value + alpha * (reward + gamma * next_max - old_value)
                q_table[state, action] = new_value

                state = next_state
                episode_rewards += reward

                iterate += 1

            rewards_per_episode.append(episode_rewards)

            if episode % 1000 == 0:
                print(f"Episode {episode} done, episode reward: {episode_rewards}, "
                      f"mean reward per iteration: {episode_rewards / iterate}, iterations: {iterate}")

    elif args.model_name == 'TD_F':
        alpha = args.alpha  # Learning rate
        gamma = args.gamma  # Discount factor
        epsilon = args.epsilon  # Exploration rate
        num_episodes = args.num_episodes
        lambd = args.td_lambda # Lambda value

        for episode in range(num_episodes):
            state = env.reset()[0]
            done = False
            episode_rewards = 0
            iterate = 0

            states = [state]
            actions = []
            rewards = []

            while not done:
                action = epsilon_greedy_policy(state, env, epsilon, q_table)  # Epsilon-greedy action selection
                next_state, reward, done, _, _ = env.step(action)
                if args.env_name == 'frozenlake':
                    if done and reward != 1:
                        reward = -10
                    else:
                        reward = -1

                states.append(next_state)
                actions.append(action)
                rewards.append(reward)

                state = next_state
                episode_rewards += reward

                iterate += 1

            T = len(states) - 1  # The number of time steps

            # Calculate returns and update Q values
            for t in range(T):
                G = 0  # Return
                W = 1  # Weight for the return
                for n in range(t, T):
                    G += gamma ** (n - t) * rewards[n]
                    if n + 1 < T:
                        G += gamma ** (n - t + 1) * np.max(q_table[states[n + 1]])
                    q_table[states[t], actions[t]] += alpha * W * (G - q_table[states[t], actions[t]])
                    W *= lambd * gamma

            rewards_per_episode.append(episode_rewards)

            if episode % 1000 == 0:
                print(f"Episode {episode} done, episode reward: {episode_rewards}, "
                      f"mean reward per iteration: {episode_rewards / iterate}, iterations: {iterate}")

    elif args.model_name == 'TD_B':
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
                if args.env_name == 'frozenlake':
                    if done and reward != 1:
                        reward = -10
                    else:
                        reward = -1

                # TD(0) update
                old_value = q_table[state, action]
                next_max = np.max(q_table[next_state])
                new_value = old_value + alpha * (reward + gamma * next_max - old_value)
                q_table[state, action] = new_value

                state = next_state
                episode_rewards += reward

                iterate += 1

            rewards_per_episode.append(episode_rewards)

            if episode % 1000 == 0:
                print(f"Episode {episode} done, episode reward: {episode_rewards}, "
                      f"mean reward per iteration: {episode_rewards / iterate}, iterations: {iterate}")

    elif args.model_name == 'SARSA':
        alpha = args.alpha  # Learning rate
        gamma = args.gamma  # Discount factor
        epsilon = args.epsilon  # Exploration rate

        num_episodes = args.num_episodes
        for episode in range(num_episodes):
            state = env.reset()[0]
            action = epsilon_greedy_policy(state, env, epsilon, q_table)
            done = False
            episode_rewards = 0
            iterate=0

            while not done:
                next_state, reward, done, _, _ = env.step(action)
                if args.env_name == 'frozenlake':
                    if done and reward != 1:
                        reward = -10
                    else:
                        reward = -1
                next_action = epsilon_greedy_policy(next_state, env, epsilon, q_table)

                # Update Q-value using the SARSA update rule
                old_value = q_table[state, action]
                next_value = q_table[next_state, next_action]
                new_value = old_value + alpha * (reward + gamma * next_value - old_value)
                q_table[state, action] = new_value

                state, action = next_state, next_action
                episode_rewards += reward
                iterate+=1

            rewards_per_episode.append(episode_rewards)
            if episode % 1000 == 0:
                print(f"Episode {episode} done, episode reward: {episode_rewards}, "
                      f"mean reward per iteration: {episode_rewards / iterate}, iterations: {iterate}")


    elif args.model_name == 'Q_learning':
        alpha = args.alpha  # Learning rate
        gamma = args.gamma  # Discount factor
        epsilon = args.epsilon  # Exploration rate

        num_episodes = args.num_episodes
        for episode in range(num_episodes):
            state = env.reset()[0]
            done = False
            episode_rewards = 0
            iterate=0
            while not done:
                action = epsilon_greedy_policy(state, env, epsilon, q_table)
                next_state, reward, done, _, _ = env.step(action)
                if args.env_name == 'frozenlake':
                    if done and reward != 1:
                        reward = -10
                    else:
                        reward = -1

                # Update Q-value using the Bellman equation
                old_value = q_table[state, action]
                next_max = np.max(q_table[next_state])
                new_value = old_value + alpha * (reward + gamma * next_max - old_value)
                q_table[state, action] = new_value

                state = next_state
                episode_rewards += reward
                iterate+=1

            rewards_per_episode.append(episode_rewards)
            if episode % 1000 == 0:
                print(f"Episode {episode} done, episode reward: {episode_rewards}, "
                      f"mean reward per iteration: {episode_rewards / iterate}, iterations: {iterate}")




    elif args.model_name == 'DQN':
        input_dim = env.observation_space.n
        output_dim = env.action_space.n

        target_dqn = DQN(input_dim, output_dim)
        behavior_dqn = DQN(input_dim, output_dim)
        behavior_dqn.load_state_dict(target_dqn.state_dict())
        memory = ReplayBuffer()

        optimizer = optim.Adam(target_dqn.parameters(), lr=args.alpha)
        #optimizer = optim.RMSprop(target_dqn.parameters(),lr=args.alpha)

        epsilon = args.epsilon
        num_episodes = args.num_episodes

        rewards_per_episode = []
        target_update_interval = 40  # Update target network every 1000 episodes

        for episode in range(num_episodes):
            state = env.reset()[0]
            state = state_preprocess(state, num_states=env.observation_space.n)
            done = False
            episode_rewards = 0
            iterate=0

            while not done:
                action = behavior_dqn.sample_action(state, epsilon,env)
                next_state, reward, done, _, _ = env.step(action)
                next_state = state_preprocess(next_state, num_states=env.observation_space.n)
                if args.env_name == 'frozenlake':
                    if done and reward != 1:
                        reward = -10
                    else:
                        reward = -1

                #next_state_one_hot = np.eye(input_dim)[next_state]
                done_mask = 0.0 if done else 1.0
                memory.put((state, action, reward, next_state, done_mask))
                state = next_state
                episode_rewards += reward
                iterate+=1



                if memory.size() > args.batch_size:
                    model_train(behavior_dqn, target_dqn, memory, optimizer, args)
                    if episode % target_update_interval == 0:
                        target_dqn.load_state_dict(behavior_dqn.state_dict())

            rewards_per_episode.append(episode_rewards)



            if episode % 100 == 0:
                print(f"Episode {episode} done, episode reward: {episode_rewards}, "
                      f"mean reward per iteration: {episode_rewards / iterate}, iterations: {iterate},epsilon : {epsilon}")

            epsilon = max(0.01, epsilon * 0.9)


    with open(f"models/{args.model_name}_{args.env_name}_q_table.pkl", "wb") as f:
        pickle.dump(q_table, f)

    return q_table, rewards_per_episode
def state_preprocess(state:int, num_states:int):
    """
    Convert an state to a tensor and basically it encodes the state into
    an onehot vector. For example, the return can be something like tensor([0,0,1,0,0])
    which could mean agent is at state 2 from total of 5 states.

    """
    onehot_vector = torch.zeros(num_states, dtype=torch.float32)
    onehot_vector[state] = 1
    return onehot_vector

class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.FC = nn.Sequential(
            nn.Linear(input_dim, 12),
            nn.ReLU(inplace=True),
            nn.Linear(12, 8),
            nn.ReLU(inplace=True),
            nn.Linear(8, output_dim)
        )

        for layer in [self.FC]:
            for module in layer:
                if isinstance(module, nn.Linear):
                    nn.init.kaiming_uniform_(module.weight, nonlinearity='relu')

    def forward(self, x):
        Q = self.FC(x)
        return Q

    def sample_action(self, state, epsilon, env):
        if np.random.rand() < epsilon:
            return env.action_space.sample()
        else:
            with torch.no_grad():
                state=torch.FloatTensor(state).unsqueeze(0)
                q_value = self.forward(state)
            return q_value.argmax().item()

def model_train(behavior_dqn, target_dqn, memory, optimizer, args):
    if memory.size() < args.batch_size:
        return

    states, actions, rewards, next_states, dones = memory.sample(args.batch_size)

    q_values = behavior_dqn(states)
    next_q_values = target_dqn(next_states)
    q_value = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
    next_q_value = next_q_values.max(1)[0]
    expected_q_value = rewards + (args.gamma * next_q_value * dones)

    loss = F.mse_loss(q_value, expected_q_value.detach())
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

class ReplayBuffer():
    def __init__(self):
        self.buffer = collections.deque(maxlen=buffer_limit)

    def put(self, transition):
        self.buffer.append(transition)

    def sample(self, batch_size):
        mini_batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*mini_batch)

        # 각 항목을 리스트로 변환 후 텐서로 변환
        states = torch.FloatTensor(np.array(states))
        actions = torch.LongTensor(np.array(actions))
        rewards = torch.FloatTensor(np.array(rewards))
        next_states = torch.FloatTensor(np.array(next_states))
        dones = torch.FloatTensor(np.array(dones))


        return states, actions, rewards, next_states, dones
    def size(self):
        return len(self.buffer)

def epsilon_greedy_policy(state, env, epsilon, q_table):
    if np.random.rand() < epsilon:
        return env.action_space.sample()  # Explore: choose a random action
    else:
        return np.argmax(q_table[state])

