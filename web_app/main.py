from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import threading
import time  # Убедитесь, что time импортируется правильно
from env.drift_car_env import DriftCarEnv
from agent.dqn_agent import DQNAgent
import numpy as np
import pygame

app = FastAPI()

app.mount("/static", StaticFiles(directory="web_app/static"), name="static")
templates = Jinja2Templates(directory="web_app/templates")

training_status = {"running": False, "episode": 0, "reward": 0}

def train_agent():
    env = DriftCarEnv()
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.shape[0]
    agent = DQNAgent(state_size, action_size)

    done = False
    batch_size = 32
    total_rewards = []

    for e in range(100):
        state = env.reset()
        state = np.reshape(state, [1, state_size])
        episode_reward = 0
        for time_step in range(100):  # Используем другое имя для переменной
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            reward = reward if not done else -10
            next_state = np.reshape(next_state, [1, state_size])
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            episode_reward += reward
            if done:
                break
            if len(agent.memory) > batch_size:
                agent.replay(batch_size)
            env.render()
        total_rewards.append(episode_reward)
        training_status["episode"] = e
        training_status["reward"] = np.mean(total_rewards[-10:])
        time.sleep(0.1)  # Задержка для обновления статуса

    env.close()
    pygame.quit()
    training_status["running"] = False

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "status": training_status})

@app.post("/start_training")
async def start_training():
    if not training_status["running"]:
        training_status["running"] = True
        threading.Thread(target=train_agent).start()
        return {"status": "training started"}
    return {"status": "training already running"}

@app.get("/status")
async def get_status():
    return training_status