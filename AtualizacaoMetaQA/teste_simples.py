from generate_paths import generate_paths
from beam_executor import execute_beam
from utils import get_embeddings, load_graph
from Model import Model

device = "cuda:0"

kg_model_path = "../Data/Graph_data/MetaQA/MetaQAProcess/new_complex_metaqa_100_inv/"
kg_model_name = "trained_model.pkl"

embedding_matrices, entity2idx, rel2idx, idx2rel = get_embeddings(
    kg_model_path,
    kg_model_name
)

idx2entity = {v:k for k,v in entity2idx.items()}

model = Model(
    embedding_matrices,
    0.1,
    True,
    True
).to(device)

nx_graph = load_graph(
    "../Data/Graph_data/MetaQA/MetaQAProcess/MetaQA_graph.pkl"
)



# carregar modelo e grafo

results = generate_paths("questions.txt")

sample = results[0]

head = "Catch Me If You Can"

topk = 5

print(sample)

result = execute_beam(
    head,
    sample["paths"],
    sample["scores"],
    model,
    entity2idx,
    idx2entity,
    rel2idx,
    nx_graph,
    device,
    topk
)

print(result)