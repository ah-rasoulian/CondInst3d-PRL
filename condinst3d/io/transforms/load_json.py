import json
from dataclasses import dataclass
from typing import Any, Dict, Hashable, Iterable, Mapping, Optional, Sequence, Union

from monai.transforms import MapTransform
from monai.config import KeysCollection


@dataclass
class LoadJSONd(MapTransform):
    """
    MONAI-style dictionary transform to load JSON files.

    Example input item:
        {"image": "case_001.nii.gz", "ann": "case_001.json"}

    After transform:
        {"image": "case_001.nii.gz", "ann": <dict parsed from json>}

    Notes:
    - Supports .json and .jsonl (JSON Lines) via `jsonl=True`.
    - If `allow_missing_keys=True`, missing keys are skipped.
    - If a value is already a dict/list (already loaded), it is left unchanged.
    """

    keys: KeysCollection
    encoding: str = "utf-8"
    jsonl: bool = False
    allow_missing_keys: bool = False
    strict: bool = True  # if False, returns None on decode errors

    def __post_init__(self):
        super().__init__(self.keys, allow_missing_keys=self.allow_missing_keys)

    def _load_json(self, path: str) -> Any:
        with open(path, "r", encoding=self.encoding) as f:
            if self.jsonl:
                # JSON lines -> list of objects
                out = []
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    out.append(json.loads(line))
                return out
            return json.load(f)

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
                        f"LoadJSONd expected a file path (str) at key '{key}', got {type(val)}"
                    )
                d[key] = None
                continue

            try:
                d[key] = self._load_json(val)
            except Exception:
                if self.strict:
                    raise
                d[key] = None

        return d
