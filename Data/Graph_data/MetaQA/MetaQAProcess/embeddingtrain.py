from pykeen.triples import TriplesFactory
from pykeen.pipeline import pipeline

tf = TriplesFactory.from_path("kb.tsv",
                              create_inverse_triples=True)

print("Triplas:", tf.num_triples)
print("Entidades:", tf.num_entities)
print("Relações:", tf.num_relations)

training, testing, validation = tf.split(
    ratios=(0.7, 0.15, 0.15),
    random_state=42
)

result = pipeline(
    training=training,
    testing=testing,
    validation=validation,
    model="ComplEx",
    model_kwargs={
        "embedding_dim": 256
    },
    training_kwargs={
        "num_epochs": 100,
        "use_tqdm_batch": True
    }
)

result.save_to_directory("new_complex_metaqa_100_inv")

print("Treinamento concluído.")

# -------------------------
# Construção do nx_graph
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

with open(
    "complex_metaqa_100_inv/MetaQA_graph.pkl",
    "wb"
) as f:

    pickle.dump(G, f)

print("Grafo salvo.")