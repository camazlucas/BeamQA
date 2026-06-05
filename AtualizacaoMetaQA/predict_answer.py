def answer_question(
    head,
    paths,
    scores,
    topk=5
):

    answer, final_score = path_finder_rec(
        head,
        paths,
        scores,
        model,
        entity2idx,
        idx2entity,
        rel2idx,
        nx_graph,
        device,
        topk=topk
    )

    return answer, final_score