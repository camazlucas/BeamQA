import pandas as pd
import networkx as nx
import pickle
from tqdm import tqdm

KB_FILE = "../Data/Graph_data/MetaQA/MetaQA/kb.tsv"
OUTPUT_FILE = "../Data/Graph_data/MetaQA/MetaQA/MetaQA_graph.pkl"

g = nx.MultiDiGraph()

df = pd.read_csv(
    KB_FILE,
    sep="\t",
    header=None,
    names=["head", "relation", "tail"]
)

print("Triplas:", len(df))

for _, row in tqdm(df.iterrows(), total=len(df)):
    g.add_edge(
        row["head"],
        row["tail"],
        data=row["relation"]
    )

with open(OUTPUT_FILE, "wb") as f:
    pickle.dump(g, f)

print("Nós:", g.number_of_nodes())
print("Arestas:", g.number_of_edges())
print("Salvo em:", OUTPUT_FILE)