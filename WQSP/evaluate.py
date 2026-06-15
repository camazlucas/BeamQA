import re
import time
import statistics

from generate_paths import generate_paths
from beamQA import path_finder_candidates
from predict_answer import predict_answer

from utils import get_embeddings, load_graph
from Model import Model
import argparse

# ==========================================================
# CONFIGURAÇÕES do Argparse
# ==========================================================
parser = argparse.ArgumentParser()

parser.add_argument(
    "--test_file",
    type=str,
    required=True
)

parser.add_argument(
    "--topk",
    type=int,
    default=20
)

parser.add_argument(
    "--output_file",
    type=str,
    default=None
)

# ==========================================================
# CONFIGURAÇÕES
# ==========================================================

args = parser.parse_args()

TEST_FILE = args.test_file

device = "cuda:0"

kg_model_path = "Data/Graph_data/FreeBase/complex_freebase_100_inv"
kg_model_name = "complex_100_inv.pkl"

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
    "../Data/Graph_data/FreeBase/freebase_nxgraph.pkl"
)


# ==========================================================
# MÉTRICAS
# ==========================================================

hits1 = 0
f1_sum = 0.0
candidate_recall_sum = 0.0
candidate_hits = 0

times = []

total_questions = 0

results_log = []


# ==========================================================
# AVALIAÇÃO
# ==========================================================

global_start = time.time()

with open(
    TEST_FILE,
    "r",
    encoding="utf-8"
) as f:

    for line_id, line in enumerate(f):

        line = line.strip()

        if not line:
            continue
        
        # ################## DEBUG ####################
        # print(repr(line))
        # break
        # #############################################

        
        try:

            question, answers, _ = line.split("\t")

        except ValueError:

            print(
                f"Linha {line_id + 1} ignorada."
            )

            continue

        # ----------------------------------
        # HEAD ENTITY
        # ----------------------------------

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

        # ----------------------------------
        # GOLD ANSWERS
        # ----------------------------------

        gold_answers = set(
            answer.strip()
            for answer in answers.split("|")
            if answer.strip()
        )

        # ----------------------------------
        # TEMPO
        # ----------------------------------

        start = time.time()

        try:

            sample = generate_paths(
                question
            )

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

            candidate_entities = {
                candidate["entity"]
                for candidate in candidates
                if candidate["entity"] is not None
            }

            candidate_recall = (
                len(candidate_entities & gold_answers)
                / len(gold_answers)
            )

            candidate_recall_sum += candidate_recall

            if len(candidate_entities & gold_answers) > 0:
                candidate_hits += 1

            # #################### DEBUG ######################

            # print("\nPergunta:")
            # print(question)

            # print("\nGold:")
            # print(gold_answers)

            # print("\nPredição:")
            # print(pred_answer)

            # break

            # ##################################################

        except Exception as e:

            print(
                f"Erro na pergunta {line_id + 1}:",
                e
            )

            pred_answer = None

        elapsed = time.time() - start

        times.append(elapsed)

        total_questions += 1

        # ----------------------------------
        # HITS@1
        # ----------------------------------

        correct = (
            pred_answer in gold_answers
        )

        if correct:
            hits1 += 1

        # ----------------------------------
        # F1
        # ----------------------------------

        pred_set = (
            {pred_answer}
            if pred_answer is not None
            else set()
        )

        tp = len(
            pred_set & gold_answers
        )

        precision = (
            tp / len(pred_set)
            if len(pred_set) > 0
            else 0
        )

        recall = (
            tp / len(gold_answers)
            if len(gold_answers) > 0
            else 0
        )

        if precision + recall > 0:

            f1 = (
                2
                * precision
                * recall
                / (precision + recall)
            )

        else:

            f1 = 0

        f1_sum += f1

        # ----------------------------------
        # LOG
        # ----------------------------------

        results_log.append(
            {
                "question": question,
                "head": head,
                "prediction": pred_answer,
                "gold": list(gold_answers),
                "correct": correct,
                "f1": f1,
                "time": elapsed
            }
        )

        if total_questions % 100 == 0:

            print(
                f"Processadas {total_questions} perguntas"
            )


# ==========================================================
# RESULTADOS FINAIS
# ==========================================================

total_time = time.time() - global_start

if total_questions == 0:
    raise RuntimeError(
        "Nenhuma pergunta foi processada."
    )

hits1_score = (
    hits1 / total_questions
)

mean_f1 = (
    f1_sum / total_questions
)

mean_candidate_recall = (
    candidate_recall_sum / total_questions
)

candidate_hits_score = (
    candidate_hits / total_questions
)

avg_time = (
    sum(times) / len(times)
)

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

print(
    "Candidate Hits:",
    round(candidate_hits_score, 4)
)

print(
    "Candidate Recall:",
    round(mean_candidate_recall, 4)
)

print(
    "F1:",
    round(mean_f1, 4)
)

print(
    "Tempo total:",
    round(total_time, 2),
    "s"
)

print(
    "Tempo medio:",
    round(avg_time, 4),
    "s"
)

print(
    "Tempo minimo:",
    round(min(times), 4),
    "s"
)

print(
    "Tempo maximo:",
    round(max(times), 4),
    "s"
)

if len(times) > 1:

    print(
        "Desvio padrao:",
        round(
            statistics.stdev(times),
            4
        ),
        "s"
    )

if args.output_file is not None:

    import csv

    with open(
        args.output_file,
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=[
                "question",
                "head",
                "prediction",
                "gold",
                "correct",
                "f1",
                "time"
            ]
        )

        writer.writeheader()

        for row in results_log:
            writer.writerow(row)