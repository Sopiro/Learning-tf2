import gym_2048
import gym
import numpy as np
import tensorflow as tf
import math
import random
import matplotlib.pyplot as plt

env = gym.make('2048-v0')

layer_count = 12  # 2^1 to 2^11, 0
table: dict = {2 ** i: i for i in range(layer_count)}


# Preprocess game state as one-hot
def preprocess(obs):
    x = np.zeros((4, 4, layer_count))

    for i in range(4):
        for j in range(4):
            if obs[i, j] > 0:
                v = min(obs[i, j], 2 ** (layer_count - 1))
                x[i, j, table[v]] = 1
            else:
                x[i, j, 0] = 1

    return x


# Define model. CNN.
def build_model():
    filters = 128

    x = tf.keras.Input(shape=(4, 4, layer_count))

    conv_a = tf.keras.layers.Conv2D(filters, kernel_size=(2, 1), activation='relu')(x)
    conv_b = tf.keras.layers.Conv2D(filters, kernel_size=(1, 2), activation='relu')(x)

    conv_aa = tf.keras.layers.Conv2D(filters, kernel_size=(2, 1), activation='relu')(conv_a)
    conv_ab = tf.keras.layers.Conv2D(filters, kernel_size=(1, 1), activation='relu')(conv_a)
    conv_ba = tf.keras.layers.Conv2D(filters, kernel_size=(2, 1), activation='relu')(conv_b)
    conv_bb = tf.keras.layers.Conv2D(filters, kernel_size=(1, 1), activation='relu')(conv_b)

    flat = [tf.keras.layers.Flatten()(a) for a in [conv_a, conv_b, conv_aa, conv_ab, conv_ba, conv_bb]]

    concat = tf.keras.layers.Concatenate()(flat)
    dense = tf.keras.layers.Dense(256, activation='relu')(concat)
    out = tf.keras.layers.Dense(4, activation='linear')(dense)

    model = tf.keras.Model(inputs=x, outputs=out)
    model.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.0005), loss='mse')
    # model.summary()
    return model


model = build_model()
target_model = build_model()

model.load_weights('../dqn_weights/dqn_weights_model')
target_model.load_weights('../dqn_weights/dqn_weights_target')

gamma = 0.9
batch_size = 512
max_memory = batch_size * 64  # batches per epoch would be 64
replay_memory = []

action_swap_array = [[0, 0, 2, 2, 1, 3, 1, 3],
                     [1, 3, 1, 3, 0, 0, 2, 2],
                     [2, 2, 0, 0, 3, 1, 3, 1],
                     [3, 1, 3, 1, 2, 2, 0, 0]]


# Append transition with image reinforcement
def append_sample(state, action, reward, next_state, done):
    g0 = state
    g1 = g0[::-1, :, :]
    g2 = g0[:, ::-1, :]
    g3 = g2[::-1, :, :]
    r0 = state.swapaxes(0, 1)
    r1 = r0[::-1, :, :]
    r2 = r0[:, ::-1, :]
    r3 = r2[::-1, :, :]

    g00 = next_state
    g10 = g00[::-1, :, :]
    g20 = g00[:, ::-1, :]
    g30 = g20[::-1, :, :]
    r00 = next_state.swapaxes(0, 1)
    r10 = r00[::-1, :, :]
    r20 = r00[:, ::-1, :]
    r30 = r20[::-1, :, :]

    states = [g0, g1, g2, g3, r0, r1, r2, r3]
    next_states = [g00, g10, g20, g30, r00, r10, r20, r30]

    for i in range(8):
        replay_memory.append([
            states[i],
            action_swap_array[action][i],
            reward,
            next_states[i],
            done
        ])


def train_model():
    np.random.shuffle(replay_memory)

    # Number of batches per epoch
    len = max_memory // batch_size

    for k in range(len):
        # batch_size * (s0, a, r, s1, d)
        mini_batch = replay_memory[k * batch_size:(k + 1) * batch_size]

        states = np.zeros((batch_size, 4, 4, layer_count))
        actions = []
        rewards = []
        next_states = np.zeros((batch_size, 4, 4, layer_count))
        dones = []

        # Accumulate all data from mini-batch to train
        for i in range(batch_size):
            states[i] = mini_batch[i][0]
            actions.append(mini_batch[i][1])
            rewards.append(mini_batch[i][2])
            next_states[i] = mini_batch[i][3]
            dones.append(mini_batch[i][4])

        # DQN Returns predicted Q value
        # batch_size * (q_left, q_up, q_right, q_down)
        answer = model.predict(states)
        target = target_model.predict(next_states)

        # Generate y-labels to train
        # Gradient descent step on loss of (y - Q(s, a|θ))^2
        for i in range(batch_size):
            # Reusing variable 'answer' as y-label
            if dones[i]:
                answer[i][actions[i]] = rewards[i]
            else:
                answer[i][actions[i]] = rewards[i] + gamma * np.amax(target[i])

        model.fit(states, answer, batch_size=batch_size, epochs=1, verbose=0)


def softmax(logits):
    exp_logits = np.exp(logits - np.amax(logits))
    sum_exp_logits = np.sum(exp_logits)
    return exp_logits / sum_exp_logits


max_episodes = 10
epsilon = 0.9
epsilon_min = 0.01

scores = []
steps = []
iteration = 0

train_count = 0

for i in range(max_episodes):
    if i % 100 == 0 and i != 0:
        print('score mean:', np.mean(scores[-100:]),
              'step mean:', np.mean(steps[-100:]),
              'iteration:', iteration,
              'epsilon:', epsilon)

    prev_obs = env.reset()

    score = 0
    step = 0
    moving_dir = np.array([1, 1, 1, 1])
    prev_max = np.max(prev_obs)

    while True:
        iteration += 1

        # Select action with ε-greedy policy
        if random.random() < epsilon:
            action = env.action_space.sample()
        else:
            x = preprocess(prev_obs)
            q_logits = model.predict(np.expand_dims(x, axis=0))[0]
            prob = softmax(q_logits)
            prob = prob * moving_dir
            action = np.argmax(prob)

        obs, reward, done, info = env.step(action)

        score += reward
        step += 1

        # Not moved
        if reward == 0 and np.array_equal(obs, prev_obs):
            moving_dir[action] = 0
            continue
        else:
            moving_dir = np.array([1, 1, 1, 1])

        # Customize reward to boost learning
        now_max = np.max(obs)
        if prev_max < now_max:
            prev_max = now_max
            reward = math.log(now_max, 2) * 0.1
        else:
            reward = 0
        reward += np.count_nonzero(prev_obs) - np.count_nonzero(obs) + 1

        # Store transition to replay memory
        append_sample(preprocess(prev_obs), action, reward, preprocess(obs), done)

        # Start train when train data fully filled
        if len(replay_memory) >= max_memory:
            train_model()
            replay_memory = []

            train_count += 1
            # Every 4 steps, Reset Q'=Q
            if train_count % 4 == 0:
                target_model.set_weights(model.get_weights())

        prev_obs = obs

        if epsilon > epsilon_min and iteration % 2500 == 0:
            epsilon = epsilon * 0.995

        if done:
            break

    scores.append(score)
    steps.append(step)

    print(i, 'score:', score, 'step:', step, 'max', np.max(obs), 'memory len:', len(replay_memory))

model.save_weights('./dqn_weights/dqn_weights_model')
target_model.save_weights('./dqn_weights/dqn_weights_target')
print(scores)

N = 100
rolling_mean = [np.mean(scores[i:i + N]) for i in range(len(scores) - N + 1)]

plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.scatter(range(len(scores)), scores, marker='.')
plt.subplot(1, 2, 2)
plt.plot(rolling_mean)
plt.show()

test_scores = []
max_tile = {}
iteration = 0
train_count = 0

# Do test
for i in range(0):
    if i % 100 == 0 and i != 0:
        print('score mean:', np.mean(scores[-100:]),
              'step mean:', np.mean(steps[-100:]),
              'iteration:', iteration,
              'epsilon:', epsilon)

    prev_obs = env.reset()
    score = 0
    step = 0
    moving_dir = np.array([1, 1, 1, 1])

    while True:
        iteration += 1

        x = preprocess(prev_obs)
        q_logits = model.predict(np.expand_dims(x, axis=0))[0]
        prob = softmax(q_logits)
        prob = prob * moving_dir
        action = np.argmax(prob)

        obs, reward, done, info = env.step(action)

        score += reward
        step += 1

        # Not moved
        if reward == 0 and np.array_equal(obs, prev_obs):
            moving_dir[action] = 0
            continue
        else:
            moving_dir = np.array([1, 1, 1, 1])

        prev_obs = obs

        if done:
            now_max = np.max(obs)
            max_tile[now_max] = max_tile.get(now_max, 0) + 1
            break

    test_scores.append(score)
    print(i, 'score:', score, 'step:', step, 'max', np.max(obs))
print(max_tile)
