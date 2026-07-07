"""Robust JSON extraction from model output (handles nested objects)."""
from __future__ import annotations

import json

_DECODER = json.JSONDecoder()


def first_json(text: str, keys: tuple[str, ...] | None = None) -> dict | None:
    """Return the first decodable JSON object in `text`, optionally requiring
    one of `keys`. Uses raw_decode so nested braces (e.g. a value that itself
    contains `{...}`) parse correctly, unlike a flat regex."""
    i = 0
    n = len(text)
    while i < n:
        j = text.find("{", i)
        if j == -1:
            return None
        try:
            obj, end = _DECODER.raw_decode(text, j)
        except json.JSONDecodeError:
            i = j + 1
            continue
        if isinstance(obj, dict) and (keys is None or any(k in obj for k in keys)):
            return obj
        i = end
    return None
