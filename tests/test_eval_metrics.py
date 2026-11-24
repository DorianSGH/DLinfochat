import pytest

from evals.metrics import Target, first_hit_rank, hit_rate, mrr


def test_target_matches_section_and_its_subsections():
    target = Target("Souls", "Haunted Souls")
    assert target.matches("Souls", "Haunted Souls")
    assert target.matches("Souls", "Haunted Souls > Soul Bag")
    assert not target.matches("Souls", "Haunted Souls Extra")  # not a subsection
    assert not target.matches("Souls", "Overview")
    assert not target.matches("Haunt", "Haunted Souls")


def test_intro_target_matches_only_the_intro():
    target = Target("Abrams", "")
    assert target.matches("Abrams", "")
    assert not target.matches("Abrams", "Weapon")


def test_page_target_matches_any_section():
    assert Target("Rejuvenator").matches("Rejuvenator", "Anything > Here")


def test_first_hit_rank():
    retrieved = [("Heroes", ""), ("Abrams", "Abilities > Siphon Life"), ("Abrams", "")]
    assert first_hit_rank(retrieved, [Target("Abrams", "Abilities > Siphon Life")]) == 2
    assert first_hit_rank(retrieved, [Target("Midboss")]) is None
    # Several acceptable targets: the earliest match wins.
    assert first_hit_rank(retrieved, [Target("Abrams", ""), Target("Heroes")]) == 1


def test_hit_rate_and_mrr():
    ranks = [1, 3, None, 7]
    assert hit_rate(ranks, k=6) == 0.5  # ranks 1 and 3
    assert hit_rate(ranks, k=10) == 0.75
    assert mrr(ranks, k=6) == pytest.approx((1 + 1 / 3) / 4)
