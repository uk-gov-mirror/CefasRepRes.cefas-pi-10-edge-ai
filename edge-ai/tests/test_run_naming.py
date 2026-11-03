import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "gui"))

from run_naming import ADJECTIVES, ANIMALS, memorable_words, timestamped_run_name


def test_memorable_words_contains_one_adjective_and_one_animal():
    adjective, animal = memorable_words().split("-")

    assert adjective in ADJECTIVES
    assert animal in ANIMALS


def test_timestamped_run_name_appends_two_memorable_words():
    run_name = timestamped_run_name("2026-10-21T08-57-00Z")

    assert run_name.startswith("2026-10-21T08-57-00Z-")
    adjective, animal = run_name.removeprefix("2026-10-21T08-57-00Z-").split("-")
    assert adjective in ADJECTIVES
    assert animal in ANIMALS
