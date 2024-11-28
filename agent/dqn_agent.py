import random
from collections import deque
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=10000)  # Увеличим размер буфера памяти
        self.gamma = 0.95  # Дисконтирующий фактор
        self.epsilon = 1.0  # Начальная вероятность исследования
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self._build_model()

    def _build_model(self):
        model = Sequential()
        model.add(Dense(128, input_dim=self.state_size, activation='relu'))  # Увеличим количество нейронов
        model.add(Dense(128, activation='relu'))  # Увеличим количество нейронов
        model.add(Dense(128, activation='relu'))  # Добавим еще один слой
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return np.random.uniform(-1, 1, self.action_size)  # Возвращаем массив действий
        act_values = self.model.predict(state)
        return act_values[0]  # Возвращаем массив действий

    def replay(self, batch_size=512, epochs=10):  # Увеличим размер батча до 512 и количество эпох до 10
        if len(self.memory) < batch_size:
            return

        minibatch = random.sample(self.memory, batch_size)
        states, actions, rewards, next_states, dones = zip(*minibatch)

        states = np.array(states).reshape(batch_size, self.state_size)
        next_states = np.array(next_states).reshape(batch_size, self.state_size)
        actions = np.array(actions).reshape(batch_size, self.action_size)
        rewards = np.array(rewards).reshape(batch_size, 1)
        dones = np.array(dones).reshape(batch_size, 1)

        dataset = tf.data.Dataset.from_tensor_slices((states, actions, rewards, next_states, dones))
        dataset = dataset.shuffle(buffer_size=1000).batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)

        for epoch in range(epochs):
            for state, action, reward, next_state, done in dataset:
                target = reward.numpy()  # Преобразуем тензор в массив NumPy
                done = done.numpy()  # Преобразуем тензор в массив NumPy
                if not done.any():  # Используем any() для проверки значений в массиве done
                    target = (reward + self.gamma * np.amax(self.model.predict(next_state)[0]))
                target_f = self.model.predict(state)
                for i in range(batch_size):
                    target_f[i][np.argmax(action[i])] = target[i]
                self.model.fit(state, target_f, epochs=1, verbose=0)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay