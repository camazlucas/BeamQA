from pykeen_loader import PyKEENLoader

loader = PyKEENLoader("complex_metaqa_100")

entity2idx, rel2idx, emb_ent, emb_rel = \
    loader.load_pykeen_checkpoint("trained_model.pkl")

print(len(entity2idx))
print(len(rel2idx))
print(len(emb_ent))
print(len(emb_rel))

print(emb_ent[0].shape)
print(emb_rel[0].shape)