import gym
import random

env = gym.make('MountainCar-v0')

# # print gym attributes
# print(env.observation_space)
# print(env.observation_space.low)
# print(env.observation_space.high)
# print()
# print(env.action_space)
# print(env._max_episode_steps)

for i in range(10):
    step = 0
    score = 0

    action = 0
    pre_obs = env.reset()

    while True:
        obs, reward, done, info = env.step(action)
        # print(observation, reward, done, info)

        if pre_obs[1] * obs[1] < 0:
            action = 0 if action == 2 else 2

        pre_obs = obs
        score += reward
        step += 1
        env.render()

        if done:
            break

    env.close()
    print('score:', score)
    print('step:', step)
