def predict_answer(candidates):

    if not candidates:
        return {
            "answer": None,
            "score": 0,
            "triples": []
        }

    max_score = float("-inf")
    best_answer = None
    best_triples = []
    best_path = None

    for candidate in candidates:

        final_score = (
            candidate["beam_score"]
            * candidate["path_score"]
        )

        if final_score > max_score:

            max_score = final_score
            best_answer = candidate["entity"]
            best_triples = candidate["triples"]
            best_path = candidate["path"]

    return {
        "answer": best_answer,
        "score": max_score,
        "triples": best_triples,
        "path": best_path
    }