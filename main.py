import argparse
import warnings
from train import train
from test import test
from arguments import set_parameters
from test_model import model_test
warnings.filterwarnings("ignore")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='q_learning', help='Name of the model') #  "MC","TD","SARSA","Q_learning","DQN"
    parser.add_argument('--env_name', type=str, default='cliffwalking', help='Name of the env')  # 'cliffwalking', 'frozenlake', 'taxi'
    parser.add_argument('--alpha', type=float, default=0.1, help='Learning rate')
    parser.add_argument('--gamma', type=float, default=0.99, help='Discount factor')
    parser.add_argument('--epsilon', type=float, default=0.5, help='Epsilon greedy rate')
    parser.add_argument('--glie', type=bool, default=True, help='Glie')
    parser.add_argument('--glie_update_time', type=int, default=10, help='glie_update_time')
    parser.add_argument('--hidden_dim', type=int, default=64, help='DQN network hidden dim size')
    parser.add_argument('--batch_size', type=int, default=32, help='DQN batch size')
    parser.add_argument('--threshold', type=int, default=40, help='DQN target network update threshold')
    parser.add_argument('--num_episodes', type=int, default=800, help='Number of training episodes')
    parser.add_argument('--plot_train_test', type=bool, default=False, help='Number of training episodes')
    args, unknown = parser.parse_known_args()


    # Replace parameter with env and model_name to set hyperparameters
    args.env_name = 'frozenlake'  # 'cliffwalking', 'frozenlake', 'taxi'
    args.model_name = 'DQN'         #  "MC","TD","SARSA","Q_learning","DQN"
    args= set_parameters(args)


    # train
    # _, rewards_per_episode, loss_per_episode,_ = train(args)
    # test(args)


    # function for test and draw graphs
    model_test(args,'model compare')


if __name__ == '__main__':
    main()
