# import pandas as pd

# print(
#     pd.read_csv(
#         "complex_metaqa/training_triples/entity_to_id.tsv.gz",
#         sep="\t"
#     ).head()
# )

# print(
#     pd.read_csv(
#         "complex_metaqa/training_triples/relation_to_id.tsv.gz",
#         sep="\t"
#     ).head()
# )

import torch

model = torch.load(
    "complex_metaqa_100/trained_model.pkl",
    map_location="cpu",
    weights_only=False
)

entity_emb = model.entity_representations[0]()
relation_emb = model.relation_representations[0]()

print(entity_emb.shape)
print(relation_emb.shape)