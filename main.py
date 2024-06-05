import argparse
import matplotlib.pyplot as plt
import warnings
from train import train
from test import test
warnings.filterwarnings("ignore")

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
    parser.add_argument('--epsilon', type=float, default=0.45, help='Exploration rate')
    parser.add_argument('--td_lambda', type=bool, default=False, help='TD lambda')
    parser.add_argument('--td_lambda_lambda', type=float, default=0.1, help='TD lambda rate')
    parser.add_argument('--td_lambda_dir', type=str, default='forward', help='TD lambda diraction')
    parser.add_argument('--batch_size', type=int, default='32', help='DQN batch size')
    parser.add_argument('--threshold', type=int, default=500, help='DQN target network update threshold')
    parser.add_argument('--num_episodes', type=int, default=3000, help='Number of training episodes')
    args, unknown = parser.parse_known_args()

    # env_list = ['cliffwalking', 'frozenlake', 'taxi']
    # model_list = ["q_learning", "sarsa" ]  # "TD" "MC" "q_learning", "sarsa" , "dqn"
    #
    # all_rewards = []
    #
    # for env in env_list:
    #     print(f"env is {env}")
    #     model_rewards_per_episode = []
    #     for model in model_list:
    #         print(f"model is {model}")
    #         args.env_name = env
    #         args.model_name = model
    #         _, rewards_per_episode = train(args)
    #         test(args)
    #         model_rewards_per_episode.append(rewards_per_episode)
    #     all_rewards.append(model_rewards_per_episode)

    args.env_name = 'cliffwalking'
    args.model_name = 'dqn'
    _, rewards_per_episode = train(args)
    test(args)

    #plot_rewards(env_list, model_list, all_rewards)


if __name__ == '__main__':
    main()
