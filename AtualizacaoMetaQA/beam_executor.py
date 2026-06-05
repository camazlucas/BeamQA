import torch

from utils import get_embeddings, load_graph
from Model import Model
from beamQA import path_finder_rec


device = "cuda:0" if torch.cuda.is_available() else "cpu"

kg_model_path = "../Data/Graph_data/MetaQA/MetaQA/complex_metaqa_100_inv/"
kg_model_name = "trained_model.pkl"

embedding_matrices, entity2idx, rel2idx, idx2rel = \
    get_embeddings(
        kg_model_path,
        kg_model_name
    )

model = Model(
    embedding_matrices,
    dropout_val=0.1,
    do_batchnorm=True,
    do_dropout=True
).to(device)

model.eval()

idx2entity = {
    v: k
    for k, v in entity2idx.items()
}

nx_graph = load_graph(
    "../Data/Graph_data/MetaQA/MetaQA/MetaQA_graph.pkl"
)


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