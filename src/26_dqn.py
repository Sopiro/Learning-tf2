import gym_2048
import gym

env = gym.make('2048-v0')
obs = env.reset()

score = 0
steps = 0

while True:
    action = env.action_space.sample()
    obs, reward, done, info = env.step(action)

    score += reward
    steps += 1
    if done:
        break

print(obs)
env.close()
print(score, steps)
