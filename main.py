import tensorflow as tf
import logging

# Настройка уровня логирования TensorFlow
tf.get_logger().setLevel(logging.ERROR)

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
    batch_size = 512
    total_rewards = []

    for e in range(5000):  # Увеличим количество эпизодов до 5000
        state = env.reset()
        state = np.reshape(state, [1, state_size])
        episode_reward = 0
        for time in range(2000):  # Увеличим количество шагов в каждом эпизоде до 2000
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            reward = reward if not done else -10
            next_state = np.reshape(next_state, [1, state_size])
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            episode_reward += reward
            if done:
                print(f"Эпизод: {e}/{5000}, счет: {time}, награда: {episode_reward}")
                break
            if len(agent.memory) > batch_size:
                agent.replay(batch_size)
            env.render()
        total_rewards.append(episode_reward)
        if e % 100 == 0:
            print(f"Средняя награда за последние 100 эпизодов: {np.mean(total_rewards[-100:])}")

    env.close()
    pygame.quit()

if __name__ == "__main__":
    main()