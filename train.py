import gymnasium as gym
import numpy as np
env = gym.make('FrozenLake-v1', map_name = "4x4", is_slippery = True, render_mode = 'human')

state = env.reset()[0]
terminated = False
truncated = False

# para el Q-Learning
q_table = np.zeros((env.observation_space.n, env.action_space.n))
gamma = 0.95
learning_rate = 0.8
episodes = 1000

# Parámetros de exploración
epsilon = 1.0
max_ep = 1.0
min_ep = 0.01
decay = 0.005 


for i in range(episodes):
    state = env.reset()
    step = 0
    finished = False
    for step in range(99):

        tradeoff = np.random.uniform(0,1)

        if tradeoff > epsilon:
            action = np.argmax(q_table[state,:])

        else:
            action = env.action_space.sample()
        
        new_state, reward, finished, _ = env.step(action)

        q_table[state, action] = q_table[state, action] + learning_rate * (reward + gamma * np.max(q_table[new_state, :]) - q_table[state, action])

        state = new_state

        if finished == True:
            break
    # decrecer epsilon por cada iteracion para que sea menos codicioso
    epsilon = min_ep + (max_ep - min_ep) * np.exp(-decay * i) 

env.reset()
rewards = []
for episode in range(100):
    state = env.reset()
    step = 0
    finished = False
    total_rewards = 0
    for step in range(99):
        action = np.argmax(q_table[state,:])
        new_state, reward, done, info = env.step(action)
        total_rewards += reward
        
        if done:
            rewards.append(total_rewards)
            break
        state = new_state

env.close()
print ("Score over time: " + str(sum(rewards)/100))