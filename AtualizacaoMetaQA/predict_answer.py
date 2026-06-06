def predict_answer(candidates):

    max_score = 0
    best_answer = None
    best_triples = None

    for candidate in candidates:

        final_score = (
            candidate["beam_score"]
            * candidate["path_score"]
        )

        if final_score > max_score:

            max_score = final_score
            best_answer = candidate["entity"]
            best_triples = candidate["triples"]

    return {
        "answer": best_answer,
        "score": max_score,
        "triples": best_triples
    }