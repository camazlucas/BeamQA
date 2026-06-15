import argparse

from pykeen.triples import TriplesFactory
from pykeen.pipeline import pipeline

parser = argparse.ArgumentParser()
parser.add_argument("--kb", type=str, required=True,
                    help="Caminho para o arquivo TSV da base de conhecimento")

parser.add_argument("--output", type=str, required=True,
                    help="Diretório onde o kg será salvo")

parser.add_argument("--embdim", type=str, required=True,
                    help="Definicao do tamanho do embedding")

args = parser.parse_args()

tf = TriplesFactory.from_path(
    args.kb,
    create_inverse_triples=False
)

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
        "embedding_dim": arg.embdim
    },
    training_kwargs={
        "num_epochs": 100,
        "use_tqdm_batch": True,
        "batch_size": 128
    }
)

result.save_to_directory(args.output)

print("Treinamento concluído.")