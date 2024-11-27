import os
from dotenv import load_dotenv

load_dotenv()

WIDTH = int(os.getenv('WIDTH'))
HEIGHT = int(os.getenv('HEIGHT'))
FPS = int(os.getenv('FPS'))