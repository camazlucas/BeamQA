from generate_paths import generate_paths
from beam_executor import execute_beam

# carregar modelo e grafo

results = generate_paths("teste.txt")

sample = results[0]

head = "ginger rogers"

topk = 5

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