"""Basic tests for the LLM fallback path (no API key required)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine.llm import ask_llm, extract_json


def test_ask_llm_returns_dict_with_text_and_source():
    result = ask_llm("system", "user prompt")
    assert isinstance(result, dict)
    assert "text" in result
    assert "source" in result
    assert result["source"] in ("llm", "fallback")


def test_extract_json_handles_bad_input():
    assert extract_json("") == {}
    assert extract_json("no json here") == {}
    assert extract_json('{"key": "value"}') == {"key": "value"}
