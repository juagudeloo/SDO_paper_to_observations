import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
current_dir = Path(__file__).parent
env_file = current_dir.parent / ".env"
load_dotenv(env_file)

# API Configuration
API_TITLE = "SDO Documents API"
API_DESCRIPTION = "API for accessing Solar Dynamics Observatory research documents extracted from the NASA ADS database."
API_VERSION = "1.0.0"

# Server Configuration
HOST = os.getenv("API_HOST", "0.0.0.0")
PORT = int(os.getenv("API_PORT", 8000))
DEBUG = os.getenv("DEBUG", "False").lower() == "true"

# Database Configuration
DATABASE_URL = os.getenv("DATABASE_URL", None)

# NASA ADS API Configuration
NASA_ADS_API_KEY = os.getenv("NASA_ADS_API_KEY", None)

# Pagination defaults
DEFAULT_PAGE_SIZE = 100
MAX_PAGE_SIZE = 1000