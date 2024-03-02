import gymnasium as gym
import numpy as np

# Environment setup with visualization
env = gym.make('FrozenLake-v1', map_name="4x4", is_slippery=True, render_mode='human')

# Initialize Q-table
q_table = np.zeros((env.observation_space.n, env.action_space.n))

# Parámetros Q-learning 
total_episodes = 100       
learning_rate = 0.8          
max_steps = 99              
gamma = 0.95                 

# Parámetros de exploración
epsilon = 1.0                
max_epsilon = 1.0            
min_epsilon = 0.01           
decay_rate = 0.005           

# Entrenamiento Q-learning 
for episode in range(total_episodes):
    # Reiniciar para cada episodio
    state = env.reset()[0]  
    finished = False

    for step in range(max_steps):
        # Elegir una acción basado en el épsilon greedy
        if np.random.uniform(0, 1) < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(q_table[state, :])

        # Tomar la acción y observar el resultado
        new_state, reward, finished, info, _ = env.step(action)  

        # Update del Q-table usando la función Q-table
        q_table[state, action] = q_table[state, action] + learning_rate * (reward + gamma * np.max(q_table[new_state, :]) - q_table[state, action])

        state = new_state

        if finished:
            break

    # Decay epsilon to reduce exploration over time
    epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-decay_rate * episode)

# Cerrar el ambiente
env.close()

# Testear el agente ya entrenado
env = gym.make('FrozenLake-v1', map_name="4x4", is_slippery=True, render_mode='human')
state = env.reset()[0]
finished = False

while not finished:
    action = np.argmax(q_table[state, :])
    state, reward, finished, info, _ = env.step(action)
    if finished:
        if reward == 1:
            print("Goal reached!")
        else:
            print("Fell into a hole.")
        break

env.close()
