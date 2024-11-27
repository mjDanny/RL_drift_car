import gym
from gym import spaces
import numpy as np
import pygame
from config.config import WIDTH, HEIGHT, FPS

class DriftCarEnv(gym.Env):
    def __init__(self):
        super(DriftCarEnv, self).__init__()

        # Определение пространства действий (управление углом поворота и скоростью)
        self.action_space = spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)

        # Определение пространства наблюдений (позиция и угол машинки)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32)

        # Начальные параметры
        self.points = [(200, 300), (600, 300)]
        self.car_position = np.array([400, 300], dtype=np.float32)
        self.car_angle = 0
        self.radius = 100
        self.speed = 0.05

        # Инициализация pygame
        pygame.init()
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("Car Drifting")
        self.clock = pygame.time.Clock()

        # Цвета
        self.WHITE = (255, 255, 255)
        self.RED = (255, 0, 0)
        self.GREEN = (0, 255, 0)

    def reset(self):
        # Сброс начальных параметров
        self.car_position = np.array([400, 300], dtype=np.float32)
        self.car_angle = 0
        return np.array([self.car_position[0], self.car_position[1], self.car_angle])

    def step(self, action):
        # Обновление угла и позиции машинки на основе действий
        self.car_angle += action[0] * self.speed
        self.car_position[0] = self.points[0][0] + self.radius * np.cos(self.car_angle)
        self.car_position[1] = self.points[0][1] + self.radius * np.sin(self.car_angle)

        # Вычисление награды (например, за близость к точкам)
        reward = 0
        for point in self.points:
            distance = np.linalg.norm(self.car_position - np.array(point))
            reward += 1 / (1 + distance)

        # Проверка условия завершения эпизода (например, если машинка выходит за пределы экрана)
        done = False
        if self.car_position[0] < 0 or self.car_position[0] > WIDTH or self.car_position[1] < 0 or self.car_position[1] > HEIGHT:
            done = True

        # Возвращаем новое состояние, награду, флаг завершения и дополнительную информацию
        return np.array([self.car_position[0], self.car_position[1], self.car_angle]), reward, done, {}

    def render(self, mode='human'):
        self.screen.fill(self.WHITE)

        # Рисуем точки
        for point in self.points:
            pygame.draw.circle(self.screen, self.RED, point, 10)

        # Рисуем машинку
        car_rect = pygame.Rect(self.car_position[0] - 10, self.car_position[1] - 5, 20, 10)
        pygame.draw.rect(self.screen, self.GREEN, car_rect)

        # Обновляем экран
        pygame.display.flip()
        self.clock.tick(FPS)  # 30 FPS