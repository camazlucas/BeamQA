from pykeen.triples import TriplesFactory
from pykeen.pipeline import pipeline

tf = TriplesFactory.from_path("kb.tsv")

print("Triplas:", tf.num_triples)
print("Entidades:", tf.num_entities)
print("Relações:", tf.num_relations)

training, testing, validation = tf.split(
    ratios=(0.8, 0.1, 0.1),
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
        "num_epochs": 5,
        "use_tqdm_batch": True
    }
)

result.save_to_directory("complex_metaqa")

print("Treinamento concluído.")