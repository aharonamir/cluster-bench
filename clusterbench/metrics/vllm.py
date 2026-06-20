"""VLLMSource — interface-conforming stub for FR-OPT.

vLLM pods are unreachable in the rev 3 topology (only LiteLLM is scrapeable).
This source exists so the metrics layer is provably pluggable: it conforms to
`MetricsSource` (AC-OPT) but always reports unreachable. If vLLM ever becomes
reachable, replace `snapshot()` with a real parser — no orchestrator or
dashboard changes required.
"""
from __future__ import annotations

from clusterbench.models import LevelDelta, ScrapeSnapshot


class VLLMSource:
    name = "vllm"

    def __init__(
        self, metrics_url: str | None, scrape_interval_s: float = 1.0
    ) -> None:
        self.metrics_url = metrics_url
        self.scrape_interval_s = scrape_interval_s

    async def snapshot(self) -> ScrapeSnapshot | None:
        # vLLM is unreachable in this topology; always None (FR-14 path).
        # When vLLM becomes reachable, parse KV-cache / scheduler queue series
        # here and return a populated ScrapeSnapshot.
        return None

    def diff(
        self,
        start: ScrapeSnapshot,
        end: ScrapeSnapshot,
        in_flight_peak: float,
        duration_s: float,
    ) -> LevelDelta:
        raise NotImplementedError(
            "VLLMSource is a stub (FR-OPT); not selectable while vLLM is "
            "unreachable. Implement when vLLM becomes scrapeable."
        )
