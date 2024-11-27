from env.drift_car_env import DriftCarEnv
from agent.dqn_agent import DQNAgent
import numpy as np
import pygame

def main():
    env = DriftCarEnv()
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.shape[0]
    agent = DQNAgent(state_size, action_size)

    # Обучение
    done = False
    batch_size = 32

    for e in range(1000):
        state = env.reset()
        state = np.reshape(state, [1, state_size])
        for time in range(500):
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            reward = reward if not done else -10
            next_state = np.reshape(next_state, [1, state_size])
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            if done:
                print(f"Эпизод: {e}/{1000}, счет: {time}")
                break
            if len(agent.memory) > batch_size:
                agent.replay(batch_size)
            env.render()

    env.close()
    pygame.quit()

if __name__ == "__main__":
    main()