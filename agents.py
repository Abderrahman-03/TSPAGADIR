"""Lightweight research and arbitration agents for the agentic tuning mode."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import requests

from tsp_core import GAConfig


@dataclass(frozen=True)
class AgentFinding:
    agent: str
    summary: str
    source_url: str
    error: str | None = None


@dataclass(frozen=True)
class AgentRecommendation:
    config: GAConfig
    rationale: str
    findings: tuple[AgentFinding, ...]


class WebResearchAgent:
    def run(self) -> AgentFinding:
        try:
            response = requests.get(
                "https://en.wikipedia.org/w/api.php",
                params={
                    "action": "query",
                    "list": "search",
                    "srsearch": "genetic algorithm traveling salesman problem",
                    "format": "json",
                },
                headers={"User-Agent": "AgadirTSPDemo/1.0 (educational application)"},
                timeout=15,
            )
            response.raise_for_status()
            results = response.json().get("query", {}).get("search", [])
            if not results:
                raise ValueError("Aucun résultat web.")
            item = results[0]
            snippet = " ".join(item.get("snippet", "").split())
            return AgentFinding(
                "Agent Web",
                f"{item.get('title', 'Résultat')} — {snippet[:420]}",
                "https://en.wikipedia.org/wiki/Genetic_algorithm",
            )
        except (requests.RequestException, ValueError, KeyError) as error:
            return AgentFinding("Agent Web", "", "", str(error))


class SemanticScholarAgent:
    def run(self) -> AgentFinding:
        try:
            response = requests.get(
                "https://api.semanticscholar.org/graph/v1/paper/search",
                params={
                    "query": "genetic algorithm traveling salesman problem",
                    "limit": 1,
                    "fields": "title,abstract,year,url",
                },
                timeout=15,
            )
            response.raise_for_status()
            results = response.json().get("data", [])
            if not results:
                raise ValueError("Aucun article Semantic Scholar trouvé.")
            paper = results[0]
            abstract = " ".join((paper.get("abstract") or "").split())
            summary = f"{paper.get('title', 'Article')}"
            if paper.get("year"):
                summary += f" ({paper['year']})"
            if abstract:
                summary += f" — {abstract[:420]}"
            return AgentFinding(summary=summary, agent="Agent Semantic Scholar", source_url=paper.get("url", ""))
        except (requests.RequestException, ValueError, KeyError) as error:
            return AgentFinding("Agent Semantic Scholar", "", "", str(error))


class ArbitrationAgent:
    """A transparent, deterministic reasoner; it does not call an LLM."""

    def run(
        self, findings: tuple[AgentFinding, ...], stop_count: int = 40
    ) -> AgentRecommendation:
        population = 100 if stop_count <= 40 else 500
        config = GAConfig(
            population_size=population,
            selection="Tournoi",
            mutation_rate=0.2,
            crossover_rate=0.8,
            generations=500,
            seed=42,
        )
        source_count = sum(not finding.error for finding in findings)
        rationale = (
            f"Pour {stop_count} points, le raisonneur retient une population de {population}, "
            "la sélection par tournoi, 0.2 de mutation, 0.8 de crossover et 500 itérations. "
            f"{source_count}/2 sources de recherche ont répondu. "
            "Ce choix est une recommandation heuristique et reste modifiable."
        )
        return AgentRecommendation(config, rationale, findings)


def run_research_agents() -> AgentRecommendation:
    with ThreadPoolExecutor(max_workers=2) as executor:
        web_future = executor.submit(WebResearchAgent().run)
        scholar_future = executor.submit(SemanticScholarAgent().run)
        findings = (web_future.result(), scholar_future.result())
    return ArbitrationAgent().run(findings)
