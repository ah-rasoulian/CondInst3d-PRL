import json
import pickle
from dataclasses import dataclass
from typing import Any, Dict, Hashable, Mapping, Optional

from monai.transforms import MapTransform
from monai.config import KeysCollection


@dataclass
class LoadInfod(MapTransform):
    """
    MONAI-style dictionary transform to load info/annotation files.

    Example input item:
        {"image": "case_001.nii.gz", "ann": "case_001.json"}
        {"image": "case_002.nii.gz", "ann": "case_002.pkl"}

    After transform:
        {"image": "case_001.nii.gz", "ann": <parsed object>}

    Notes:
    - Supports .json and .jsonl (JSON Lines) via `jsonl=True`.
    - Supports .pkl via pickle.
    - If `allow_missing_keys=True`, missing keys are skipped.
    - If a value is already a dict/list (already loaded), it is left unchanged.
    """

    keys: KeysCollection
    encoding: str = "utf-8"
    jsonl: bool = False
    allow_missing_keys: bool = False
    strict: bool = True  # if False, returns None on decode/load errors
    file_type: Optional[str] = None  # "json" | "jsonl" | "pkl" | None (auto by extension)

    def __post_init__(self):
        super().__init__(self.keys, allow_missing_keys=self.allow_missing_keys)

    def _infer_file_type(self, path: str) -> str:
        p = path.lower()
        if p.endswith(".jsonl"):
            return "jsonl"
        if p.endswith(".json"):
            return "json"
        if p.endswith(".pkl") or p.endswith(".pickle"):
            return "pkl"
        raise ValueError(
            f"LoadInfod could not infer file type from extension: {path}. "
            f"Supported: .json, .jsonl, .pkl/.pickle (or set file_type=...)."
        )

    def _load_json(self, path: str) -> Any:
        with open(path, "r", encoding=self.encoding) as f:
            if self.jsonl:
                out = []
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    out.append(json.loads(line))
                return out
            return json.load(f)

    def _load_jsonl(self, path: str) -> Any:
        # explicit jsonl loader (independent from self.jsonl)
        with open(path, "r", encoding=self.encoding) as f:
            out = []
            for line in f:
                line = line.strip()
                if not line:
                    continue
                out.append(json.loads(line))
            return out

    def _load_pkl(self, path: str) -> Any:
        with open(path, "rb") as f:
            return pickle.load(f)

    def _load_any(self, path: str) -> Any:
        ft = (self.file_type or self._infer_file_type(path)).lower()

        if ft == "json":
            return self._load_json(path)
        if ft == "jsonl":
            return self._load_jsonl(path) if not self.jsonl else self._load_json(path)
        if ft == "pkl":
            return self._load_pkl(path)

        raise ValueError(
            f"LoadInfod: unsupported file_type={self.file_type!r}. "
            f"Use 'json', 'jsonl', or 'pkl', or leave None for auto-detect."
        )

    def __call__(self, data: Mapping[Hashable, Any]) -> Dict[Hashable, Any]:
        d: Dict[Hashable, Any] = dict(data)

        for key in self.key_iterator(d):
            val = d.get(key, None)

            # already loaded
            if isinstance(val, (dict, list)):
                continue

            # must be a string path
            if not isinstance(val, str):
                if self.strict:
                    raise TypeError(
                        f"LoadInfod expected a file path (str) at key '{key}', got {type(val)}"
                    )
                d[key] = None
                continue

            try:
                d[key] = self._load_any(val)
            except Exception:
                if self.strict:
                    raise
                d[key] = None

        return d
