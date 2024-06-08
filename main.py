import argparse
import warnings
from train import train
from test import test
from plot_test import performance_comparison_plot
from arguments import set_parameters
warnings.filterwarnings("ignore")


def train_performance_comparison(args,model_list):
    env_list = ['cliffwalking', 'frozenlake', 'taxi']
    plot_result = {env: {model: [] for model in model_list} for env in env_list}

    for env in env_list:
        for model in model_list:
            args.env_name = env
            args.model_name = model
            args = set_parameters(args)
            _, rewards_per_episode = train(args)
            test(args)
            plot_result[env][model] = rewards_per_episode

    performance_comparison_plot(env_list, model_list, plot_result)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='q_learning', help='Name of the model')
    parser.add_argument('--env_name', type=str, default='cliffwalking', help='Name of the env')
    parser.add_argument('--alpha', type=float, default=0.1, help='Learning rate')
    parser.add_argument('--gamma', type=float, default=0.99, help='Discount factor')
    parser.add_argument('--epsilon', type=float, default=0.5, help='Exploration rate')
    parser.add_argument('--glie', type=bool, default=True, help='MC glie')
    parser.add_argument('--glie_update_time', type=int, default=10, help='glie_update_time')
    parser.add_argument('--batch_size', type=int, default=16, help='DQN batch size')
    parser.add_argument('--threshold', type=int, default=500, help='DQN target network update threshold')
    parser.add_argument('--num_episodes', type=int, default=500, help='Number of training episodes')
    args, unknown = parser.parse_known_args()

    # env_list = ['cliffwalking', 'frozenlake', 'taxi']
    # model_list = ["Q_learning", "TD" ]  #  "MC", "TD", "Q_learning", "SARSA" , "DQN"
    args.env_name = 'cliffwalking'
    args.model_name = 'DQN'
    args= set_parameters(args)
    _, rewards_per_episode = train(args)
    test(args)




    # MC_TD running rate test
    # model_list=["Q_learning","SARSA"]
    # train_performance_comparison(args,model_list)






if __name__ == '__main__':
    main()
