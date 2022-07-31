import os
from dotenv import load_dotenv


load_dotenv()
LOG_DIR = os.getenv('LOG_DIR')
R_PATH = os.getenv('R_PATH')
