import os
from dotenv import load_dotenv
from dataclasses import dataclass, fields

from src.utils.env_utils import getenv
from src.utils.log_utils import get_logger

# Logger setup
LOGGER = get_logger("env_utils")

# Load environment file
load_dotenv()


@dataclass
class Settings:
    DATA_DIR: str = getenv("DATA_DIR", "./data")
    RAW_DIR: str = DATA_DIR + getenv("RAW_DIR", "./data/raw")
    PROCESSED_DIR: str = DATA_DIR + getenv("PROCESSED_DIR", "./data/processed")
    PROFILES_DIR: str = DATA_DIR + getenv("PROFILES_DIR", "./data/profiles")
    FIGURES_DIR: str = DATA_DIR + getenv("FIGURES_DIR", "./data/figures")


SETTINGS = Settings()

for field in fields(Settings):
    name = field.name
    value = getattr(SETTINGS, name)
    LOGGER.info(f"{name}: {value}")
    if name.endswith("_DIR"):
        os.makedirs(value, exist_ok=True)
