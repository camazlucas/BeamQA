from pykeen.triples import TriplesFactory
import networkx as nx
import pickle
import argparse
import os

# -------------------------
# ARGPARSE
# -------------------------

parser = argparse.ArgumentParser()

parser.add_argument(
    "--kg_file",
    type=str,
    required=True
)

parser.add_argument(
    "--output_dir",
    type=str,
    required=True
)

args = parser.parse_args()

# -------------------------
# CARREGAR KG
# -------------------------

tf = TriplesFactory.from_path(
    args.kg_file,
    create_inverse_triples=True
)

# -------------------------
# CONSTRUÇÃO DO GRAFO
# -------------------------

G = nx.DiGraph()

id_to_entity = {
    v: k
    for k, v in tf.entity_to_id.items()
}

id_to_relation = {
    v: k
    for k, v in tf.relation_to_id.items()
}

for h, r, t in tf.mapped_triples.tolist():

    head = id_to_entity[h]
    relation = id_to_relation[r]
    tail = id_to_entity[t]

    G.add_edge(
        head,
        tail,
        data=relation
    )

    G.add_edge(
        tail,
        head,
        data=relation + "_inv"
    )

# -------------------------
# SALVAR
# -------------------------

os.makedirs(
    args.output_dir,
    exist_ok=True
)

graph_file = os.path.join(
    args.output_dir,
    "MetaQA_graph.pkl"
)

with open(
    graph_file,
    "wb"
) as f:

    pickle.dump(G, f)

print(
    f"Grafo salvo em: {graph_file}"
)