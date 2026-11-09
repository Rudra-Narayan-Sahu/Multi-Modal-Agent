import json
import os
import threading
from typing import Dict, Any, Optional


class LongTermMemoryStore:
    """
    Persistent store for user preferences, custom research context,
    and cross-session memory.
    """

    def __init__(self, storage_path: str = "processed_data/long_term_memory.json"):
        self.storage_path = storage_path
        self._lock = threading.Lock()
        self._data: Dict[str, Dict[str, Any]] = {}
        self._load()

    def _load(self):
        if os.path.exists(self.storage_path):
            try:
                with open(self.storage_path, "r", encoding="utf-8") as f:
                    self._data = json.load(f)
            except Exception:
                self._data = {}

    def _save(self):
        os.makedirs(os.path.dirname(self.storage_path), exist_ok=True)
        try:
            with open(self.storage_path, "w", encoding="utf-8") as f:
                json.dump(self._data, f, indent=2)
        except Exception:
            pass

    def get_user_preferences(self, user_id: str) -> Dict[str, Any]:
        with self._lock:
            return self._data.get(user_id, {}).get("preferences", {})

    def set_user_preference(self, user_id: str, key: str, value: Any) -> None:
        with self._lock:
            if user_id not in self._data:
                self._data[user_id] = {"preferences": {}, "notes": []}
            self._data[user_id]["preferences"][key] = value
            self._save()

    def add_research_note(self, user_id: str, note: str) -> None:
        with self._lock:
            if user_id not in self._data:
                self._data[user_id] = {"preferences": {}, "notes": []}
            self._data[user_id]["notes"].append(note)
            self._save()


long_term_memory = LongTermMemoryStore()

