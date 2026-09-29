"""Research Orchestrator – plans and validates novel research questions."""

from __future__ import annotations

import json
from typing import Any, Dict, List

from engine import llm
from engine.paper_ingestion import search_papers
from engine.research_knowledge import get_knowledge
from engine.gap_detector import get_detector
from engine.methodology_extractor import extract_methodology

RESEARCH_SYSTEM = (
    "You are a research planner for astrobiology research. "
    "Given a research question, analyze it and return valid JSON with these fields:\n"
    "- question_type: one of atmospheric, population, formation, comparative, habitability, stellar, other\n"
    "- data_needed: list of data sources required to answer the question\n"
    "- analysis_steps: ordered list of steps to perform the analysis\n"
    "- novelty_angle: what would be new or novel about this research\n"
    "- expected_paper_type: one of research_article, review, letter\n"
    "Return ONLY valid JSON. No markdown fences."
)


class ResearchOrchestrator:
    """Orchestrates the research pipeline from question to validated plan."""

    def __init__(self) -> None:
        self.knowledge = get_knowledge()
        self.gap_detector = get_detector()

    def plan_research(self, question: str) -> Dict[str, Any]:
        result = llm.ask_llm(RESEARCH_SYSTEM, question, temperature=0.2, max_tokens=800)
        response = result["text"]
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            return {"question_type": "other", "data_needed": [],
                    "analysis_steps": [], "novelty_angle": "",
                    "expected_paper_type": "research_article"}

    def search_literature(self, question: str) -> List[Dict[str, Any]]:
        return search_papers(question)

    def find_gaps(self, question: str, papers: List[Dict[str, Any]]) -> List[str]:
        gaps = self.gap_detector.detect_gaps_for_question(question, papers)
        return [g.get("question", "") for g in gaps]

    def check_novelty(self, plan: Dict[str, Any],
                      existing_papers: List[Dict[str, Any]]) -> Dict[str, Any]:
        existing_topics = [p.get("title", "") for p in existing_papers[:20]]
        prompt = (f"Research plan: {json.dumps(plan)}\n"
                  f"Existing papers: {json.dumps(existing_topics)}\n"
                  "Is this research novel? Return JSON with 'is_novel' (bool) "
                  "and 'reasoning' (str).")
        result = llm.ask_llm("You are a novelty assessor.", prompt, temperature=0.2, max_tokens=500)
        response = result["text"]
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            return {"is_novel": True, "reasoning": "Could not parse assessment."}

    def run_research(self, question: str) -> Dict[str, Any]:
        plan = self.plan_research(question)
        papers = self.search_literature(question)
        gaps = self.find_gaps(question, papers)
        novelty = self.check_novelty(plan, papers)
        return {
            "plan": plan,
            "existing_papers": papers,
            "gaps": gaps,
            "novelty": novelty,
        }

    def generate_research_context(self, question: str) -> str:
        result = self.run_research(question)
        lines = [
            f"Question: {question}",
            f"Type: {result['plan'].get('question_type', 'other')}",
            f"Papers found: {len(result['existing_papers'])}",
            f"Gaps identified: {len(result['gaps'])}",
            "Gaps: " + "; ".join(result["gaps"][:5]),
            f"Novel: {result['novelty'].get('is_novel', True)}",
            f"Reasoning: {result['novelty'].get('reasoning', '')}",
        ]
        return "\n".join(lines)


_orchestrator: ResearchOrchestrator | None = None


def get_orchestrator() -> ResearchOrchestrator:
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = ResearchOrchestrator()
    return _orchestrator
