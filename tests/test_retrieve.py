from app.rag.retrieve import reciprocal_rank_fusion, rerank_order, section_url


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


def test_rerank_order_only_reorders_within_depth():
    ranking = [1, 2, 3, 4]
    scores = {1: 0.1, 2: 0.9, 3: 0.5, 4: 1.0}
    # 4 has the best score but sits beyond the depth, so the reranker never saw it.
    assert rerank_order(ranking, scores, depth=3) == [2, 3, 1, 4]


def test_rerank_order_keeps_fused_order_for_ties():
    assert rerank_order([5, 6, 7], {5: 0.5, 6: 0.5, 7: 0.9}, depth=3) == [7, 5, 6]


def test_section_url_points_at_heading_anchor():
    url = "https://deadlock.wiki/Abrams"
    assert section_url(url, "Abilities > Siphon Life") == f"{url}#Siphon_Life"
    assert section_url(url, "") == url
