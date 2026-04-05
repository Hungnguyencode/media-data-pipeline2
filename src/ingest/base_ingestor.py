from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict


class BaseIngestor(ABC):
    @abstractmethod
    def ingest(self, source: str) -> Dict[str, Any]:
        raise NotImplementedError