# with open(r"D:\KGQAArmazenamento\DATASET METAQA\kb.txt", "r", encoding="utf-8") as fin, \
#      open("kb.tsv", "w", encoding="utf-8") as fout:

#     for line in fin:
#         fout.write(line.replace("|", "\t"))

import torch

model = torch.load(
    "complex_metaqa/trained_model.pkl",
    map_location="cpu",
    weights_only=False
)

print(model)
print(type(model.entity_representations))
print(type(model.relation_representations))

entity_emb = model.entity_representations[0]()
relation_emb = model.relation_representations[0]()

print(entity_emb.shape)
print(relation_emb.shape)