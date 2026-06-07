from __future__ import annotations

from typing import Iterable, Optional, Tuple


def levenshtein_distance(a: str, b: str) -> int:
    """Compute the Levenshtein edit distance between two strings.

    Uses two rolling rows to keep the memory footprint at ``O(min(|a|, |b|))``.
    """

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
    """Find the candidate with the smallest edit distance to ``pattern``.

    Returns a tuple ``(nearest, distance)``. When the candidate iterable is
    empty, ``(None, -1)`` is returned. Ties are broken deterministically by
    lexicographic order, which keeps the explanation module reproducible.
    """

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
