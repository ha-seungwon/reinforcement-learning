# plot_utils.py

import numpy as np
import matplotlib.pyplot as plt

def plot_rewards(reward_history, args):
    sma = np.convolve(reward_history, np.ones(50) / 50, mode='valid')

    plt.figure()
    plt.title("Obtained Rewards")
    plt.plot(reward_history, label='Raw Reward', color='#F1BD81', alpha=1)
    plt.plot(sma, label='SMA 50', color='#2B6C6D')
    plt.xlabel("Episode")
    plt.ylabel("Rewards")
    plt.legend()

    if args.num_episodes:
        plt.savefig(f'docs/{args.model_name}_{args.env_name}_reward_plot.png', format='png', dpi=600, bbox_inches='tight')
    plt.tight_layout()
    plt.grid(True)
    plt.show()
    plt.clf()
    plt.close()



def plot_loss(loss_history, args):
    plt.figure()
    plt.title("Network Loss")
    plt.plot(loss_history, label='Loss', color='#8549f2', alpha=1)
    plt.xlabel("Episode")
    plt.ylabel("Loss")

    if args.num_episodes:
        plt.savefig(f'docs/{args.model_name}_{args.env_name}_Loss_plot.png', format='png', dpi=600, bbox_inches='tight')
    plt.tight_layout()
    plt.grid(True)
    plt.show()
    plt.clf()
    plt.close()

def plot_rewards_all_models(all_rewards, args):
    model_colors = {
        "MC": "#FF4500",          # OrangeRed
        "TD": "#1E90FF",          # DodgerBlue
        "SARSA": "#FF4500",       # ForestGreen ##228B22
        "Q_learning": "#1E90FF",  # DarkGoldenRod FFB800
        "DQN": "#FF4500"
    }
    env_styles = {
        "cliffwalking": "-",
        "frozenlake": "-",
        "taxi": "-"
    }

    plt.figure()
    plt.title("Obtained Rewards for All Models and Environments")

    for env, model_rewards in all_rewards.items():
        for model, rewards in model_rewards.items():
            plt.plot(rewards, label=f'{model} - {env} Raw', color=model_colors[model], alpha=0.5, linestyle=env_styles[env])

    plt.xlabel("Episode")
    plt.ylabel("Rewards")
    plt.legend()
    plt.tight_layout()
    plt.grid(True)
    plt.savefig(f'docs/{args.model_name}_{args.env_name}_reward_plot_all_models.png', format='png', dpi=600, bbox_inches='tight')
    plt.show()
    plt.clf()
    plt.close()


def plot_rewards_mc_td_lr(all_rewards, args):
    model_colors = {
        "MC": "#FF4500",  # OrangeRed
        "TD": "#1E90FF"  # DodgerBlue
    }
    lr_colors = {
        1: "#FF6347",  # Tomato
        0.01: "#4682B4",  # SteelBlue
        0.05: "#32CD32"  # LimeGreen
    }
    lr_styles = {
        1: "-",
        0.01: "--",
        0.05: ":"
    }

    for env, model_rewards in all_rewards.items():
        for model in model_rewards.keys():
            plt.figure()
            plt.title(f"Obtained Rewards for {model} in {env} with Different Learning Rates")

            for lr, rewards in model_rewards[model].items():
                plt.plot(rewards, label=f'LR {lr} Raw', color=lr_colors[lr], alpha=0.5, linestyle=lr_styles[lr])

            plt.xlabel("Episode")
            plt.ylabel("Rewards")
            plt.legend()
            plt.tight_layout()
            plt.grid(True)
            plt.savefig(f'docs/{env}_{model}_reward_plot_different_lr.png', format='png', dpi=600, bbox_inches='tight')
            plt.show()
            plt.clf()
            plt.close()


def plot_rewards_mc_glie_models(all_rewards, args):
    glie_colors = {
        True: "#FF6347",    # Tomato
        False: "#4682B4"   # SteelBlue
    }

    plt.figure()
    plt.title(f"Obtained Rewards for MC with GLIE vs Non-GLIE")

    for glie, rewards in all_rewards.items():
        label_prefix = "GLIE" if glie else "Non-GLIE"
        plt.plot(rewards, label=f'{label_prefix} Raw', color=glie_colors[glie], alpha=0.5)

    plt.xlabel("Episode")
    plt.ylabel("Rewards")
    plt.legend()
    plt.tight_layout()
    plt.grid(True)
    plt.savefig(f'docs/{args.model_name}_{args.env_name}_reward_plot_glie_vs_non_glie.png', format='png', dpi=600, bbox_inches='tight')
    plt.show()
    plt.clf()
    plt.close()