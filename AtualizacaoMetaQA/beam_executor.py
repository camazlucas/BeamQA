from beamQA import path_finder_rec

def execute_beam(
    head,
    paths,
    score_paths,
    model,
    entity2idx,
    idx2entity,
    rel2idx,
    nx_graph,
    device=device,
    topk=topk
):

    answer, score, triples = path_finder_rec(
        head,
        paths,
        score_paths,
        model,
        entity2idx,
        idx2entity,
        rel2idx,
        nx_graph,
        device=device,
        topk=topk
    )

    return {
        "answer": answer,
        "score": score,
        "triples": triples
    }