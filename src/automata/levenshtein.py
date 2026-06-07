from __future__ import annotations

from typing import Iterable, Optional, Tuple


def levenshtein_distance(a: str, b: str) -> int:

    if a == b:
        return 0
    if len(a) < len(b):
        a, b = b, a
    if not b:
        return len(a)

    previous = list(range(len(b) + 1))
    current = [0] * (len(b) + 1)
    for i, ch_a in enumerate(a, start=1):
        current[0] = i
        for j, ch_b in enumerate(b, start=1):
            insert_cost = current[j - 1] + 1
            delete_cost = previous[j] + 1
            substitute_cost = previous[j - 1] + (0 if ch_a == ch_b else 1)
            current[j] = min(insert_cost, delete_cost, substitute_cost)
        previous, current = current, previous
    return previous[-1]


def find_nearest_pattern(
    pattern: str, candidates: Iterable[str]
) -> Tuple[Optional[str], int]:

    best: Optional[str] = None
    best_distance = -1
    for candidate in candidates:
        distance = levenshtein_distance(pattern, candidate)
        if best is None or distance < best_distance or (
            distance == best_distance and candidate < (best or "")
        ):
            best = candidate
            best_distance = distance
    return best, best_distance
