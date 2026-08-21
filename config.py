import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
CACHE_DIR = DATA_DIR / "cache"
VECTOR_INDEX = DATA_DIR / "papers.index"
VECTOR_META = DATA_DIR / "papers_meta.json"
REPORTS_DIR = DATA_DIR / "reports"

for p in (DATA_DIR, CACHE_DIR, REPORTS_DIR):
    p.mkdir(parents=True, exist_ok=True)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
CEREBRAS_API_KEY = os.getenv("CEREBRAS_API_KEY", "")
CEREBRAS_BASE_URL = os.getenv("CEREBRAS_BASE_URL", "https://api.cerebras.ai/v1")
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "cerebras")
LLM_MODEL = os.getenv("LLM_MODEL", "llama-3.3-70b")

MAST_CACHE_DAYS = int(os.getenv("MAST_CACHE_DAYS", "7"))
