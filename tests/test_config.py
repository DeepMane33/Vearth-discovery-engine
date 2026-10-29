"""Basic tests for config module."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import BASE_DIR, DATA_DIR, CACHE_DIR, REPORTS_DIR, LLM_PROVIDER, LLM_MODEL


def test_base_dir_exists():
    assert BASE_DIR.exists()


def test_data_dirs_created():
    assert DATA_DIR.exists()
    assert CACHE_DIR.exists()
    assert REPORTS_DIR.exists()


def test_llm_defaults():
    assert LLM_PROVIDER in ("cerebras", "openai", "anthropic", "")
    assert isinstance(LLM_MODEL, str)
