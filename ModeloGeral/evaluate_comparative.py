import re
import ast
import pandas as pd
import argparse

from beamQA import path_finder_candidates
from predict_answer import predict_answer

from utils import get_embeddings, load_graph
from Model import Model


# ==========================================================
# ARGPARSE
# ==========================================================

parser = argparse.ArgumentParser()

parser.add_argument(
    "--test_file",
    type=str,
    required=True
)

parser.add_argument(
    "--paths_file",
    type=str,
    required=True
)

parser.add_argument(
    "--topk",
    type=int,
    default=10
)

args = parser.parse_args()

# ==========================================================
# CONFIGURAÇÕES
# ==========================================================

device = "cuda:0"

kg_model_path = (
    "../Data/Graph_data/MetaQA/"
    "MetaQAProcess/new_complex_metaqa_100_inv/"
)

kg_model_name = "trained_model.pkl"

topk = args.topk

# ==========================================================
# CARREGAR MODELO
# ==========================================================

embedding_matrices, entity2idx, rel2idx, idx2rel = get_embeddings(
    kg_model_path,
    kg_model_name
)

idx2entity = {
    v: k
    for k, v in entity2idx.items()
}

model = Model(
    embedding_matrices,
    0.1,
    True,
    True
).to(device)

nx_graph = load_graph(
    "../Data/Graph_data/MetaQA/"
    "MetaQAProcess/new_complex_metaqa_100_inv/"
    "MetaQA_graph.pkl"
)

# ==========================================================
# CARREGAR CAMINHOS PRÉ-GERADOS
# ==========================================================

df_paths = pd.read_csv(
    args.paths_file,
    index_col=0,
    delimiter="\t"
)

df_paths = df_paths.values

# ==========================================================
# MÉTRICAS
# ==========================================================

hits1 = 0
total_questions = 0

# ==========================================================
# AVALIAÇÃO
# ==========================================================

with open(
    args.test_file,
    "r",
    encoding="utf-8"
) as f:

    for line_id, line in enumerate(f):

        line = line.strip()

        if not line:
            continue

        try:

            question, answers, _ = line.split("\t")

        except ValueError:

            print(
                f"Linha {line_id + 1} ignorada."
            )

            continue

        match = re.search(
            r"\[(.*?)\]",
            question
        )

        if match is None:

            print(
                f"Linha {line_id + 1}: entidade não encontrada."
            )

            continue

        head = match.group(1)

        gold_answers = set(
            answer.strip()
            for answer in answers.split("|")
            if answer.strip()
        )

        # --------------------------------------------------
        # CAMINHOS PRÉ-GERADOS
        # --------------------------------------------------

        beam_paths = (
            df_paths[line_id - 1][1]
            .split("|")
        )

        scores = ast.literal_eval(
            df_paths[line_id - 1][2]
        )

        scores = [
            float(score)
            for score in scores
        ]

        sample = {
            "question": question,
            "paths": beam_paths,
            "scores": scores
        }

        try:

            candidates = path_finder_candidates(
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

            prediction = predict_answer(
                candidates
            )

            pred_answer = prediction["answer"]

        except Exception as e:

            print(
                f"Erro na pergunta {line_id + 1}:",
                e
            )

            pred_answer = None

        if pred_answer in gold_answers:
            hits1 += 1

        total_questions += 1

        if total_questions % 100 == 0:

            print(
                f"Processadas {total_questions} perguntas"
            )

# ==========================================================
# RESULTADOS
# ==========================================================

if total_questions == 0:

    raise RuntimeError(
        "Nenhuma pergunta foi processada."
    )

hits1_score = hits1 / total_questions

print("\n========================")
print("RESULTADOS")
print("========================")

print(
    "Perguntas:",
    total_questions
)

print(
    "Hits@1:",
    round(hits1_score, 4)
)