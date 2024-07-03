import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import pickle

#"6x6z": [
#   "SFHFPP",
#   "FFFFHP",
#   "FHFFFF",
#   "FPHFHH",
#   "FHPFHP",
#   "GHHFPP",
#   ]

def run(episodes,is_training=True,  render=False):

    
    #env = gym.make('FrozenLake-v1', map_name="8x8",desc=generate_random_map(size=8),is_slippery=False, render_mode='human' if render else None)
    #env = gym.make('FrozenLake-v1', map_name="8x8", is_slippery=False, render_mode='human' if render else None)
    env = gym.make('FrozenLake-v1', desc= ["SFHFPP","FFFFHP","FHFFFF","FPHFHH","FHPFHP","GHHFPP",], is_slippery=False, render_mode='human' if render else None)


    if(is_training):
        q = np.zeros((env.observation_space.n, env.action_space.n)) # Q-Table
    else:
        f = open('frozen_lake8x8.pkl', 'rb')
        q = pickle.load(f)
        f.close()

    learning_rate_a = 0.9 #learning rate
    discount_factor_g = 0.9 # gamma

    epsilon = 1 # porcentagem das ações aleatorias
    epsilon_decay_rate = 0.0001 # taxa de decaimento
    rng = np.random.default_rng() # numero aleatorio

    rewards_per_episode = np.zeros(episodes)

    for i in range(episodes):
        state = env.reset()[0] # states: 0 to 63, 0=top left corner,63=bottom right corner

        terminated = False # True when fall in hole or reached goal
        truncated = False # True when actions > 200


        while(not terminated and not truncated):
            if is_training and rng.random() < epsilon:
                action = env.action_space.sample() # actions: 0=left,1=down,2=right, 3=up
            else:
                action = np.argmax(q[state,:])

            new_state, reward, terminated, truncated,_= env.step(action)
            
            if is_training: 
                q[state,action] = q[state,action] + learning_rate_a * (reward + discount_factor_g * np.max(q[new_state,:]) - q[state,action])
            
            state = new_state

        epsilon = max(epsilon - epsilon_decay_rate, 0)

        if(epsilon == 0):
            learning_rate_a = 0.0001

        if reward == 1:
            rewards_per_episode[i] = 1

    env.close()

    sum_rewards = np.zeros(episodes)
    for t in range(episodes):
        sum_rewards[t] = np.sum(rewards_per_episode[max(0,t-100):(t+1)])
    plt.plot(sum_rewards)
    plt.savefig('frozen_lake8x8.png')

    if is_training: 
        f = open("frozen_lake8x8.pkl","wb")
        pickle.dump(q,f)
        f.close()

if __name__ == '__main__':
    #run(10001)

    run(1, is_training=False, render=True) 