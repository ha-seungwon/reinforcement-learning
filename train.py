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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train(args):
    print("Training start")

    # Initialize the environment
    if args.env_name == 'frozenlake':
        env = gym.make('FrozenLake-v1', desc=None, map_name="4x4", is_slippery=False, max_episode_steps=400)
    elif args.env_name == 'cliffwalking':
        env = gym.make('CliffWalking-v0', max_episode_steps=300)
    elif args.env_name == 'taxi':
        env = gym.make('Taxi-v3', max_episode_steps=200)

    print(args, env.observation_space.n)
    env.reset()
    env.action_space.seed(42)

    # Initialize Q-table with zeros
    q_table = np.zeros((env.observation_space.n, env.action_space.n))
    rewards_per_episode = []
    loss_per_episode = []
    optimal_policy=None

    if args.model_name == 'MC':
        train_mc(args, env, q_table, rewards_per_episode)
    elif args.model_name == 'TD':
        train_td(args, env, q_table, rewards_per_episode)
    elif args.model_name == 'TD_F':
        train_td_f(args, env, q_table, rewards_per_episode)
    elif args.model_name == 'TD_B':
        train_td_b(args, env, q_table, rewards_per_episode)
    elif args.model_name == 'SARSA':
        train_sarsa(args, env, q_table, rewards_per_episode)
    elif args.model_name == 'Q_learning':
        train_q_learning(args, env, q_table, rewards_per_episode)
    elif args.model_name == 'DQN':
        optimal_policy,q_table=train_dqn(args, env, rewards_per_episode,loss_per_episode)

    # Save the Q-table or DQN model
    if args.model_name != 'DQN':
        print("model save")
        with open(f"models/{args.model_name}_{args.env_name}_q_table.pkl", "wb") as f:
            pickle.dump(q_table, f)
        #optimal_policy = extract_policy(q_table) # print final policy of env
        print("Final Optimal Policy:")




    return q_table, rewards_per_episode, loss_per_episode,optimal_policy


def extract_policy(q_table):
    optimal_policy = {}
    for state in range(q_table.shape[0]):
        optimal_action = np.argmax(q_table[state])
        optimal_policy[state] = optimal_action
    return optimal_policy
def extract_policy_dqn(model, env,args):
    optimal_policy = {}
    for state in range(env.observation_space.n):
        int_state=state
        if args.model_name =='DQN':
            state = one_hot_encode(state, env.observation_space.n)
        state_tensor = torch.tensor([state], dtype=torch.float32).to(device)  # Convert state to tensor
        q_values = model(state_tensor)  # Get Q-values from the model
        optimal_action = torch.argmax(q_values).item()  # Get the action with the highest Q-value
        optimal_policy[int_state] = optimal_action
    return optimal_policy

def train_mc(args, env, q_table, rewards_per_episode):
    N = np.zeros((env.observation_space.n, env.action_space.n))  # Visit count for each state-action pair
    k = 0
    epsilon = 1 if args.glie else args.epsilon

    print("epsilon", epsilon)
    for episode in range(args.num_episodes):
        episode_rewards = 0  # Rewards received in this episode
        history = []
        state = env.reset()[0]
        done = False
        iterate = 0
        early_stop=False

        while not done and not early_stop:
            action = epsilon_greedy_policy(state, env, epsilon, q_table)  # Epsilon-greedy action selection
            next_state, reward, done, early_stop, _ = env.step(action)
            if args.env_name == 'frozenlake':
                reward = -10 if done and reward != 1 else -1
            elif args.env_name == 'cliffwalking':
                if done and reward != -100:
                    reward = 10

            history.append((state, action, reward))
            state = next_state
            episode_rewards += reward
            iterate += 1

        # Update Q-values
        G = 0
        for t in range(len(history) - 1, -1, -1):
            state, action, reward = history[t]
            G = args.gamma * G + reward
            N[state][action] += 1
            alpha_t = args.alpha
            q_table[state][action] += alpha_t * (G - q_table[state][action])

        if args.glie and episode % args.glie_update_time == 0:
            k += 1
            epsilon = 1 / k

        rewards_per_episode.append(episode_rewards)
        if episode % 1000 == 0:
            print(
                f"Episode {episode} done, episode reward: {episode_rewards}, mean reward per iteration: {episode_rewards / iterate}, iterations: {iterate}, epsilon: {epsilon}")


def train_td(args, env, q_table, rewards_per_episode):
    alpha = args.alpha  # Learning rate
    gamma = args.gamma  # Discount factor
    epsilon = args.epsilon  # Exploration rate
    num_episodes = args.num_episodes
    k = 0
    flag = True
    first_optimal = 0

    for episode in range(num_episodes):
        state = env.reset()[0]
        done = False
        episode_rewards = 0
        iterate = 0
        early_stop = False
        while not done:
            action = epsilon_greedy_policy(state, env, epsilon, q_table)  # Epsilon-greedy action selection
            next_state, reward, done, early_stop, _ = env.step(action)
            if args.env_name == 'frozenlake':
                reward = -10 if done and reward != 1 else -1
            elif args.env_name == 'cliffwalking':
                if done and reward != -100:
                    reward = 10

            # TD(0) update
            old_value = q_table[state, action]
            next_max = np.max(q_table[next_state])
            new_value = old_value + alpha * (reward + gamma * next_max - old_value)
            q_table[state, action] = new_value

            state = next_state
            episode_rewards += reward
            iterate += 1
        if episode_rewards> -6.5 and flag:
            print(episode_rewards)
            first_optimal=episode
            flag=False
            print("first_optimal",first_optimal)

        rewards_per_episode.append(episode_rewards)


        if args.glie and episode % args.glie_update_time == 0:
            k += 1
            epsilon = 1 / k
        if episode % 1000 == 0:
            print(
                f"Episode {episode} done, episode reward: {episode_rewards}, mean reward per iteration: {episode_rewards / iterate}, iterations: {iterate}")


def train_td_f(args, env, q_table, rewards_per_episode):
    alpha = args.alpha  # Learning rate
    gamma = args.gamma  # Discount factor
    epsilon = args.epsilon  # Exploration rate
    num_episodes = args.num_episodes
    lambd = args.td_lambda  # Lambda value

    for episode in range(num_episodes):
        state = env.reset()[0]
        done = False
        early_stop = False

        episode_rewards = 0
        iterate = 0
        states, actions, rewards = [state], [], []

        while not done and not early_stop:
            action = epsilon_greedy_policy(state, env, epsilon, q_table)  # Epsilon-greedy action selection
            next_state, reward, done, early_stop, _ = env.step(action)
            if args.env_name == 'frozenlake':
                reward = -10 if done and reward != 1 else -1
            elif args.env_name == 'cliffwalking':
                if done and reward != -100:
                    reward = 10

            states.append(next_state)
            actions.append(action)
            rewards.append(reward)

            state = next_state
            episode_rewards += reward
            iterate += 1

        T = len(states) - 1  # The number of time steps
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
            print(
                f"Episode {episode} done, episode reward: {episode_rewards}, mean reward per iteration: {episode_rewards / iterate}, iterations: {iterate}")


def train_td_b(args, env, q_table, rewards_per_episode):
    alpha = args.alpha  # Learning rate
    gamma = args.gamma  # Discount factor
    epsilon = args.epsilon  # Exploration rate
    num_episodes = args.num_episodes

    for episode in range(num_episodes):
        state = env.reset()[0]
        done = False
        early_stop = False
        episode_rewards = 0
        iterate = 0
        while not done and not early_stop:
            action = epsilon_greedy_policy(state, env, epsilon, q_table)  # Epsilon-greedy action selection
            next_state, reward, done, early_stop, _ = env.step(action)
            if args.env_name == 'frozenlake':
                reward = -10 if done and reward != 1 else -1
            elif args.env_name == 'cliffwalking':
                if done and reward != -100:
                    reward = 10

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
            print(
                f"Episode {episode} done, episode reward: {episode_rewards}, mean reward per iteration: {episode_rewards / iterate}, iterations: {iterate}")


def train_sarsa(args, env, q_table, rewards_per_episode):
    alpha = args.alpha  # Learning rate
    gamma = args.gamma  # Discount factor
    epsilon = args.epsilon  # Exploration rate
    num_episodes = args.num_episodes

    for episode in range(num_episodes):
        state = env.reset()[0]
        action = epsilon_greedy_policy(state, env, epsilon, q_table)
        done = False
        early_stop = False
        episode_rewards = 0
        iterate = 0

        while not done and not early_stop:
            next_state, reward, done, early_stop, _ = env.step(action)
            if args.env_name == 'frozenlake':
                reward = -10 if done and reward != 1 else -1
            elif args.env_name == 'cliffwalking':
                if done and reward != -100:
                    reward = 10
            next_action = epsilon_greedy_policy(next_state, env, epsilon, q_table)

            old_value = q_table[state, action]
            new_value = old_value + alpha * (reward + gamma * q_table[next_state, next_action] - old_value)
            q_table[state, action] = new_value

            state, action = next_state, next_action
            episode_rewards += reward
            iterate += 1

        rewards_per_episode.append(episode_rewards)
        if episode % 1000 == 0:
            print(
                f"Episode {episode} done, episode reward: {episode_rewards}, mean reward per iteration: {episode_rewards / iterate}, iterations: {iterate}")


def train_q_learning(args, env, q_table, rewards_per_episode):
    alpha = args.alpha  # Learning rate
    gamma = args.gamma  # Discount factor
    epsilon = args.epsilon  # Exploration rate
    num_episodes = args.num_episodes

    for episode in range(num_episodes):
        state = env.reset()[0]
        done = False
        early_stop = False
        episode_rewards = 0
        iterate = 0

        while not done and not early_stop:
            action = epsilon_greedy_policy(state, env, epsilon, q_table)
            next_state, reward, done, early_stop, _ = env.step(action)
            if args.env_name == 'frozenlake':
                reward = -10 if done and reward != 1 else -1
            elif args.env_name == 'cliffwalking':
                if done and reward != -100:
                    reward = 10

            old_value = q_table[state, action]
            next_max = np.max(q_table[next_state])
            new_value = old_value + alpha * (reward + gamma * next_max - old_value)
            q_table[state, action] = new_value

            state = next_state
            episode_rewards += reward
            iterate += 1

        rewards_per_episode.append(episode_rewards)
        if episode % 1000 == 0:
            print(
                f"Episode {episode} done, episode reward: {episode_rewards}, mean reward per iteration: {episode_rewards / iterate}, iterations: {iterate}")


def train_dqn(args, env, rewards_per_episode,loss_per_episode):
    print("dqn train start")
    state_dim=env.observation_space.n
    behavior_model = DQN(state_dim, env.action_space.n,args).to(device)
    optimizer = optim.Adam(behavior_model.parameters(), lr=args.alpha)
    epsilon = args.epsilon
    memory = ReplayBuffer(10000)
    num_episodes = args.num_episodes
    target_model = DQN(state_dim, env.action_space.n,args).to(device)
    target_model.load_state_dict(behavior_model.state_dict())

    first_optimal = 0
    flag=True
    for episode in range(num_episodes):
        state = env.reset()[0]
        state = one_hot_encode(state, state_dim)
        done = False
        episode_rewards = 0
        episode_loss = 0
        early_stop = False
        iterate = 0


        while not done and not early_stop:
            action = epsilon_greedy_policy_dqn(state, env, epsilon, behavior_model)
            next_state, reward, done, early_stop, _ = env.step(action)
            next_state = one_hot_encode(next_state, state_dim)
            if args.env_name == 'frozenlake':
                reward = -10 if done and reward != 1 else -1
            elif args.env_name == 'cliffwalking':
                if done and reward != -100:
                    reward = 10

            memory.push(state, action, reward, next_state, done)
            state = next_state
            episode_rewards += reward
            iterate += 1

            if len(memory) > args.batch_size:
                loss = optimize_model(args, memory, behavior_model, target_model, optimizer)
                episode_loss+=loss
                # if done:
                #     print(
                #         f"Episode {episode} done, episode reward: {episode_rewards}, mean reward per iteration: {episode_rewards / iterate}, iterations: {iterate}, epsilon: {epsilon}, loss: {loss}")
        if episode_rewards > -7 and flag:
            first_optimal=episode
            flag= False
            print("first_optimal",first_optimal)

        if episode % args.threshold == 0:
            target_model.load_state_dict(behavior_model.state_dict())

        rewards_per_episode.append(episode_rewards)
        loss_per_episode.append(episode_loss)

        epsilon = max(0.01, epsilon * 0.995)
        if episode % 1 == 0:
            print(
                f"Episode {episode} done, episode reward: {episode_rewards}, mean reward per iteration: {episode_rewards / iterate}, iterations: {iterate}")

    torch.save(behavior_model.state_dict(), f"models/{args.model_name}_{args.env_name}_dqn_model.pt")

    return extract_policy_dqn(behavior_model,env,args),behavior_model


def epsilon_greedy_policy(state, env, epsilon, q_table):
    if random.uniform(0, 1) < epsilon:
        return env.action_space.sample()
    else:
        return np.argmax(q_table[state])


def epsilon_greedy_policy_dqn(state, env, epsilon, model):
    if random.uniform(0, 1) < epsilon:
        return env.action_space.sample()
    else:
        with torch.no_grad():
            state = torch.tensor([state], device=device, dtype=torch.float32)
            q_values = model(state)
            return q_values.argmax().item()


class DQN(nn.Module):
    def __init__(self, num_inputs, num_outputs,args):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(num_inputs, args.hidden_dim)
        self.fc2 = nn.Linear(args.hidden_dim, args.hidden_dim)
        self.fc3 = nn.Linear(args.hidden_dim, num_outputs)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        state = np.expand_dims(state, 0)
        next_state = np.expand_dims(next_state, 0)
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        state, action, reward, next_state, done = zip(*random.sample(self.buffer, batch_size))
        return np.concatenate(state), action, reward, np.concatenate(next_state), done

    def __len__(self):
        return len(self.buffer)
def one_hot_encode(state, state_dim):
    one_hot = np.zeros(state_dim)
    one_hot[state] = 1
    return one_hot


def optimize_model(args, memory, model, target_model, optimizer):
    if len(memory) < args.batch_size:
        return None

    batch = memory.sample(args.batch_size)
    state_batch = torch.tensor(batch[0], device=device, dtype=torch.float32)
    action_batch = torch.tensor(batch[1], device=device, dtype=torch.int64)
    reward_batch = torch.tensor(batch[2], device=device, dtype=torch.float32)
    next_state_batch = torch.tensor(batch[3], device=device, dtype=torch.float32)
    done_batch = torch.tensor(batch[4], device=device, dtype=torch.float32)

    q_values = model(state_batch).gather(1, action_batch.unsqueeze(1)).squeeze(1)
    next_q_values = target_model(next_state_batch).max(1)[0]
    expected_q_values = reward_batch + args.gamma * next_q_values * (1 - done_batch)

    loss = F.mse_loss(q_values, expected_q_values)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()