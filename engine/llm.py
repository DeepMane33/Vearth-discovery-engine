"""LLM client with provider support (Cerebras / OpenAI / Anthropic) and a
deterministic rule-based fallback so the full pipeline runs even without a key."""
import json
import re

try:
    from openai import OpenAI
    _HAS_OPENAI = True
except Exception:
    _HAS_OPENAI = False

try:
    import anthropic
    _HAS_ANTHROPIC = True
except Exception:
    _HAS_ANTHROPIC = False

from config import (
    LLM_PROVIDER, LLM_MODEL, OPENAI_API_KEY, ANTHROPIC_API_KEY,
    CEREBRAS_API_KEY, CEREBRAS_BASE_URL,
)


def _build_client():
    """Return a callable completion function or None if no provider configured."""
    provider = (LLM_PROVIDER or "").lower()

    if provider in ("cerebras", "openai") and _HAS_OPENAI:
        key = CEREBRAS_API_KEY if provider == "cerebras" else OPENAI_API_KEY
        base = CEREBRAS_BASE_URL if provider == "cerebras" else None
        if key:
            kwargs = {"api_key": key}
            if base:
                kwargs["base_url"] = base
            client = OpenAI(**kwargs)

            def _complete(system, user, temperature=0.3, max_tokens=1200):
                resp = client.chat.completions.create(
                    model=LLM_MODEL,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                return resp.choices[0].message.content
            return _complete

    if provider == "anthropic" and _HAS_ANTHROPIC and ANTHROPIC_API_KEY:
        client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

        def _complete(system, user, temperature=0.3, max_tokens=1200):
            msg = client.messages.create(
                model=LLM_MODEL,
                system=system,
                max_tokens=max_tokens,
                temperature=temperature,
                messages=[{"role": "user", "content": user}],
            )
            return msg.content[0].text
        return _complete

    return None


_complete_fn = _build_client()


def llm_available() -> bool:
    return _complete_fn is not None


def ask_llm(system_prompt: str, user_prompt: str, temperature: float = 0.3,
            max_tokens: int = 1200) -> dict:
    """Call the LLM and return {'text': ..., 'source': 'llm'|'fallback'}."""
    if _complete_fn is not None:
        try:
            text = _complete_fn(system_prompt, user_prompt, temperature, max_tokens)
            return {"text": text.strip(), "source": "llm"}
        except Exception as e:
            return {"text": _fallback(system_prompt, user_prompt, str(e)),
                    "source": "fallback", "error": str(e)}
    return {"text": _fallback(system_prompt, user_prompt, "no llm configured"),
            "source": "fallback"}


def _fallback(system_prompt: str, user_prompt: str, reason: str) -> str:
    """Deterministic rule-based response so the pipeline never hard-fails."""
    p = user_prompt.lower()
    if "hypothesis" in p or "anomaly" in p:
        return json.dumps({
            "hypothesis": (
                "The dip pattern is consistent with a single-body transit "
                "of moderate depth. A periodic transit-like signal is a "
                "plausible working hypothesis pending vetting."),
            "self_questions": [
                "Could this be an instrumental systematic from detector roll?",
                "Could background contamination produce a similar dip?",
                "Is the period consistent with a known eclipsing binary?",
            ],
            "confidence": "low",
            "veto": False,
            "reason": "fallback engine (no live LLM): " + reason,
        }, indent=2)
    if "paper" in p or "abstract" in p or "report" in p:
        return ("FALLBACK REPORT (no live LLM connected).\n\n"
                "A periodic transit-like anomaly was detected in the light "
                "curve and passed preliminary vetting. The dip depth and "
                "periodicity suggest a candidate worth follow-up. Exact "
                "figures are reported in the analysis section above. "
                "(Reason: " + reason + ")")
    return "fallback: " + reason


def extract_json(text: str) -> dict:
    """Best-effort extraction of a JSON object from an LLM response."""
    if not text:
        return {}
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            pass
    try:
        return json.loads(text)
    except Exception:
        return {}
