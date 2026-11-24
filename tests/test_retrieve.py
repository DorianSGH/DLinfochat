from app.rag.retrieve import reciprocal_rank_fusion, section_url


def test_rrf_prefers_items_found_by_both_methods():
    semantic = [1, 2, 3]
    keyword = [3, 1, 4]
    fused = [item for item, _ in reciprocal_rank_fusion([semantic, keyword], k=60)]
    # 2 is ranked higher than 3 by semantic search alone, but 3 appears in both lists.
    assert fused == [1, 3, 2, 4]


def test_rrf_score_is_sum_of_reciprocal_ranks():
    [(item, score)] = reciprocal_rank_fusion([[7], [7]], k=60)
    assert item == 7
    assert score == 2 / 61


def test_section_url_points_at_heading_anchor():
    url = "https://deadlock.wiki/Abrams"
    assert section_url(url, "Abilities > Siphon Life") == f"{url}#Siphon_Life"
    assert section_url(url, "") == url
