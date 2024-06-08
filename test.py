import pickle
import gym
import numpy as np


def test(args, num_episodes=1):
    print("test start")
    if args.env_name == 'frozenlake':
        env = gym.make('FrozenLake-v1', desc=None, map_name="4x4", is_slippery=False,render_mode='human') #render_mode='human',
    elif args.env_name == 'cliffwalking':
        env = gym.make('CliffWalking-v0',render_mode='human')#, render_mode='human')
    elif args.env_name == 'taxi':
        env = gym.make('Taxi-v3',render_mode='human')#, render_mode='human')

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
            state, reward, done, _, _ = env.step(action)

            episode_rewards += reward
            steps += 1

            if done:
                print("done!!",reward)
                if args.env_name =='cliffwalking':
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

