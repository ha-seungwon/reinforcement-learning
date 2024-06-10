import gym

from train import train
from test import test
from arguments import set_parameters
from plot_utils import plot_rewards, plot_loss,plot_rewards_all_models ,plot_rewards_mc_td_lr, plot_rewards_mc_glie_models # 여기서 임포트
import numpy as np
import time
import psutil
import torch
import matplotlib.pyplot as plt
def print_policy(optimal_policy, shape):
    policy_grid = np.zeros(shape, dtype=str)
    actions = ['^', '>', 'v', '<']

    for state, action in optimal_policy.items():
        row, col = divmod(state, shape[1])
        policy_grid[row][col] = actions[action]

    for row in policy_grid:
        print(' '.join(row))
def get_memory_usage():
    process = psutil.Process()
    return process.memory_info().rss / (1024 ** 2)  # 메가바이트 단위로 반환


def model_test(args,test_name):
    model_list = ["Q_learning", "TD", "SARSA", "MC", "DQN"]
    if test_name == "optimal policy test":
        ################ optimal policy test ######################
        args.env_name = 'frozenlake' #frozenlake  cliffwalking
        for model in model_list:
            args.model_name =model# 'MC'
            args = set_parameters(args)
            q_table, rewards_per_episode, loss_per_episode,optimal_policy = train(args)
            # 0: move up
            # 1: move right
            # 2: move down
            # 3: move left
            print_policy(optimal_policy, (4, 4))  # 4,12 cliff 4,4 frozen
            test(args)

    elif test_name == "Glie":
        ################ Glie test  ######################
        args.env_name = 'taxi'
        args.model_name = 'MC'

        glie_list = [True, False]
        all_rewards = {}

        for glie in glie_list:
            args = set_parameters(args)
            args.glie = glie
            if glie:
                args.epsilon = 1

            q_table, rewards_per_episode, loss_per_episode,optimal_policy = train(args)
            #test(args)
            all_rewards[glie] = rewards_per_episode

        plot_rewards_mc_glie_models(all_rewards, args)

    elif  test_name == "model compare":
        ################### 모델 성능 단순 비교 ####################
        env_list = ['frozenlake']  #cliffwalking  frozenlake taxi
        model_list = ["TD","Q_learning","DQN"]# "MC", "TD" "SARSA", "Q_learning", "DQN"]

        all_rewards = {env: {} for env in env_list}
        for env in env_list:
            for model in model_list:
                args.env_name = env
                args.model_name = model
                args = set_parameters(args)
                _, rewards_per_episode, loss_per_episode = train(args)
                #test(args)
                all_rewards[env][model] = rewards_per_episode
                # if model == 'DQN':
                #     plot_loss(loss_per_episode, args)
        plot_rewards_all_models(all_rewards,args)


    elif  test_name == 'lr':
        ################## learning rate test ###################
        env_list = ['frozenlake']#, 'frozenlake', 'taxi']
        model_list = ["MC", "TD"]
        learning_rates = [1, 0.01, 0.05]
        all_rewards = {env: {model: {} for model in model_list} for env in env_list}
        for env in env_list:
            for model in model_list:
                for lr in learning_rates:
                    args.env_name = env
                    args.model_name = model
                    args = set_parameters(args)
                    args.alpha = lr

                    _, rewards_per_episode, loss_per_episode = train(args)
                    if model not in all_rewards[env]:
                        all_rewards[env][model] = {}
                    all_rewards[env][model][lr] = rewards_per_episode
        plot_rewards_mc_td_lr(all_rewards, args)

    elif test_name =='dqn_test':
        #env = gym.make('CliffWalking-v0', max_episode_steps=300)
        env = gym.make('Taxi-v3', max_episode_steps=200)
        #env = gym.make('FrozenLake-v1', desc=None, map_name="4x4", is_slippery=False)
        start_time = time.time()
        initial_memory = get_memory_usage()
        args.env_name = 'frozenlake'
        args.model_name = 'TD'
        args = set_parameters(args)
        q_learning_q_table, q_learning_rewards, _, _ = train(args)
        q_learning_train_time = time.time() - start_time
        q_learning_memory_usage = get_memory_usage() - initial_memory


        start_time = time.time()
        initial_memory = get_memory_usage()
        args.env_name = 'frozenlake'
        args.model_name = 'DQN'
        args = set_parameters(args)
        dqn_model, dqn_rewards, _, _ = train(args)
        dqn_train_time = time.time() - start_time
        dqn_memory_usage = get_memory_usage() - initial_memory


        print(f'Q-learning training time: {q_learning_train_time:.2f} seconds')
        print(f'Q-learning memory usage: {q_learning_memory_usage:.2f} MB')

        print(f'DQN training time: {dqn_train_time:.2f} seconds')
        print(f'DQN memory usage: {dqn_memory_usage:.2f} MB')



def test_q_learning(env, q_table, num_episodes=100):
    total_rewards = []
    for episode in range(num_episodes):
        state,_ = env.reset()
        done = False
        episode_rewards = 0
        while not done:
            action = np.argmax(q_table[state])
            next_state, reward, done, _,_ = env.step(action)
            episode_rewards += reward
            state = next_state
        total_rewards.append(episode_rewards)
    return np.mean(total_rewards), total_rewards
def one_hot_encode(state, state_dim):
    one_hot = np.zeros(state_dim)
    one_hot[state] = 1
    return one_hot
# DQN 테스트
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def test_dqn(env, model, num_episodes=100):
    total_rewards = []
    for episode in range(num_episodes):
        state,_ = env.reset()
        state = one_hot_encode(state, env.observation_space.n)
        done = False
        episode_rewards = 0
        while not done:
            state_tensor = torch.tensor([state], dtype=torch.float32).to(device)
            q_values = model(state_tensor)
            action = torch.argmax(q_values).item()
            next_state, reward, done, _,_ = env.step(action)
            next_state = one_hot_encode(next_state, env.observation_space.n)
            episode_rewards += reward
            state = next_state
        total_rewards.append(episode_rewards)
    return np.mean(total_rewards), total_rewards
