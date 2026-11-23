from __future__ import annotations
from abc import ABC, abstractmethod
from ..schemas import Source
class SourceProvider(ABC):
    name: str = "base"
    @abstractmethod
    def search(self, query: str, limit: int = 8) -> list[Source]:
        """Return up to `limit` sources for `query`. May raise — callers
        (the aggregator) are responsible for catching and logging."""
        raise NotImplementedError
    def search_bounded(self, query: str, limit: int = 8, *, deadline: float | None = None, start: int = 0) -> list[Source]:
        """Optional hard-deadline search contract.
        Providers that perform network retries/throttling should override this
        method. The default preserves compatibility for deterministic or
        injected providers used by tests and legacy workflows.
        """
        return self.search(query, limit)
