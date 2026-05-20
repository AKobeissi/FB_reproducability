"""Reciprocal Rank Fusion for combining BM25 + dense results."""


def reciprocal_rank_fusion(
    result_lists: list[list[dict]],
    score_keys: list[str],
    k: int = 60,
    top_k: int = 100,
) -> list[dict]:
    """
    Fuse multiple ranked lists using RRF.
    result_lists[i] is a list of page dicts ranked by score_keys[i].
    Returns a merged list sorted by RRF score descending.
    """
    rrf_scores: dict[str, float] = {}
    candidate_map: dict[str, dict] = {}

    for results in result_lists:
        for rank, item in enumerate(results):
            cid = item["candidate_id"]
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + 1.0 / (k + rank + 1)
            if cid not in candidate_map:
                candidate_map[cid] = item

    merged = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    output = []
    for rank, (cid, rrf_score) in enumerate(merged[:top_k]):
        item = dict(candidate_map[cid])
        item["rrf_score"] = rrf_score
        item["rank"] = rank + 1
        item["retriever"] = "rrf"
        output.append(item)
    return output
