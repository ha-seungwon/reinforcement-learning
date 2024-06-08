import matplotlib.pyplot as plt


def performance_comparison_plot(env_list, model_list, plot_result):
    colors = {model_list[0]: 'b', model_list[1]: 'r'}  # Define different colors for the models

    for env in env_list:
        plt.figure(figsize=(10, 7))
        plt.title(f'Performance Comparison in {env}', fontsize=16)

        for model in model_list:
            rewards = plot_result[env][model]
            plt.plot(rewards, label=f'{model}', alpha=0.6, color=colors[model])

        plt.xlabel('Episodes')
        plt.ylabel('Rewards')
        plt.legend()
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig(f"docs/{env}_{model_list[0]}_{model_list[1]}.png")
        plt.close()