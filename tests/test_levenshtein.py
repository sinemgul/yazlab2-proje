"""Unit tests for the Levenshtein implementation and pattern matching."""

import pytest

from src.automata.levenshtein import find_nearest_pattern, levenshtein_distance


@pytest.mark.parametrize(
    "a, b, expected",
    [
        ("", "", 0),
        ("a", "", 1),
        ("", "abc", 3),
        ("abc", "abc", 0),
        ("abc", "abd", 1),
        ("kitten", "sitting", 3),
        ("flaw", "lawn", 2),
        ("abcdef", "azced", 3),
    ],
)
def test_levenshtein_distance(a: str, b: str, expected: int) -> None:
    assert levenshtein_distance(a, b) == expected


def test_levenshtein_symmetry() -> None:
    assert levenshtein_distance("abcd", "xycd") == levenshtein_distance("xycd", "abcd")


def test_find_nearest_pattern_basic() -> None:
    # "abd" and "abcd" both have edit distance 1 from "abc"; the deterministic
    # lexicographic tie-break prefers "abcd" because 'c' < 'd' at position 2.
    nearest, distance = find_nearest_pattern("abc", ["abd", "xyz", "abcd"])
    assert nearest == "abcd"
    assert distance == 1


def test_find_nearest_pattern_unique_minimum() -> None:
    nearest, distance = find_nearest_pattern("abc", ["abd", "xyz"])
    assert nearest == "abd"
    assert distance == 1


def test_find_nearest_pattern_tie_break_lexicographic() -> None:
    # Both "abe" and "abd" are at distance 1 from "abc"; lexicographic order
    # prefers "abd".
    nearest, distance = find_nearest_pattern("abc", ["abe", "abd"])
    assert nearest == "abd"
    assert distance == 1


def test_find_nearest_pattern_empty_candidates() -> None:
    nearest, distance = find_nearest_pattern("abc", [])
    assert nearest is None
    assert distance == -1
