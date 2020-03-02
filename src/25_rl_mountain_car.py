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

step = 0
score = 0
env.reset()

while True:
    action = env.action_space.sample()
    observation, reward, done, info = env.step(action)
    print(observation, reward, done, info)

    score += reward
    step += 1
    env.render()

    if done:
        break

env.close()
print('score:', score)
print('step:', step)
