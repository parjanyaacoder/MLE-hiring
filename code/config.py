import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Project Paths
CODE_DIR = Path(__file__).resolve().parent
REPO_ROOT = CODE_DIR.parent
DATA_DIR = REPO_ROOT / "data"
SUPPORT_TICKETS_DIR = REPO_ROOT / "support_tickets"
INPUT_CSV = SUPPORT_TICKETS_DIR / "support_tickets.csv"
OUTPUT_CSV = SUPPORT_TICKETS_DIR / "output.csv"
API_SPECS_FILE = DATA_DIR / "api_specs" / "internal_tools.json"

# API Settings
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")

# Expected CSV output headers
EXPECTED_HEADERS = [
    "issue", "subject", "company", "response", "product_area",
    "status", "request_type", "justification", "confidence_score",
    "source_documents", "risk_level", "pii_detected", "language",
    "actions_taken"
]

# LLM Pinned Parameters
DEFAULT_MODEL = "models/gemini-3.5-flash"
DEFAULT_TEMPERATURE = 0.0
DEFAULT_SEED = 42

# Concurrency & Retry Settings (Optimized for Free Tier rate limit of 15 RPM)
MAX_WORKERS = 1  # Use 1 worker to space out requests, increase to 10+ for paid tier
GEMINI_RETRY_COUNT = 5
GEMINI_RETRY_DELAY = 15  # Base sleep time on rate limit in seconds

# Ensure reproducibility
import random
try:
    import numpy as np
    np.random.seed(DEFAULT_SEED)
except ImportError:
    pass
random.seed(DEFAULT_SEED)
