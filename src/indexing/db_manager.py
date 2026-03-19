from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import chromadb
from chromadb.config import Settings

from src.utils import get_config, get_data_path

logger = logging.getLogger(__name__)


class DBManager:
    """Manage ChromaDB operations in a config-consistent way."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or get_config()
        vector_db_dir = self.config["paths"].get("vector_db_dir", "data/vector_db")
        self.persist_directory = Path(get_data_path(vector_db_dir))
        self.persist_directory.mkdir(parents=True, exist_ok=True)

        self.client = chromadb.PersistentClient(
            path=str(self.persist_directory),
            settings=Settings(anonymized_telemetry=False),
        )

        logger.info("DB Manager initialized with persist dir: %s", self.persist_directory)

    def list_collections(self) -> List[str]:
        return [col.name for col in self.client.list_collections()]

    def get_collection(self, name: str):
        try:
            return self.client.get_collection(name=name)
        except Exception as e:
            logger.error("Collection '%s' not found: %s", name, e)
            return None

    def delete_collection(self, name: str) -> bool:
        try:
            self.client.delete_collection(name=name)
            logger.info("Deleted collection: %s", name)
            return True
        except Exception as e:
            logger.error("Failed to delete collection '%s': %s", name, e)
            return False

    def export_collection(self, collection_name: str, output_file: str) -> None:
        collection = self.get_collection(collection_name)
        if not collection:
            return

        results = collection.get(limit=10000)
        output_data = {
            "collection_name": collection_name,
            "count": len(results.get("ids", [])),
            "data": [],
        }

        for i in range(len(results.get("ids", []))):
            output_data["data"].append(
                {
                    "id": results["ids"][i],
                    "document": results["documents"][i],
                    "metadata": results["metadatas"][i],
                }
            )

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)

        logger.info("Exported %d items to %s", len(results.get("ids", [])), output_file)

    def get_stats(self) -> Dict[str, Any]:
        collections = self.list_collections()
        return {
            "persist_directory": str(self.persist_directory),
            "total_collections": len(collections),
            "collections": collections,
        }