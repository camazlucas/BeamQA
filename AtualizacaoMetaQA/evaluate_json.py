import re
import json
import argparse

from generate_paths import (
    generate_paths,
    load_bart_model,
    METAQA_RELATIONS
)

from beamQA import path_finder_candidates

from utils import (
    get_embeddings,
    load_graph
)

from Model import Model


# ==========================================================
# ARGUMENTOS
# ==========================================================

parser = argparse.ArgumentParser()

parser.add_argument(
    "--test_file",
    type=str,
    required=True
)

parser.add_argument(
    "--output_file",
    type=str,
    required=True
)

parser.add_argument(
    "--bart_model_path",
    type=str,
    required=True
)

parser.add_argument(
    "--kg_model_path",
    type=str,
    required=True
)

parser.add_argument(
    "--kg_model_name",
    type=str,
    default="trained_model.pkl"
)

parser.add_argument(
    "--graph_file",
    type=str,
    required=True
)

parser.add_argument(
    "--topk",
    type=int,
    default=20
)

args = parser.parse_args()

device = "cuda:0"


# ==========================================================
# CARREGAR BART
# ==========================================================

print("Carregando BART...")

tokenizer, path_model = load_bart_model(
    args.bart_model_path
)


# ==========================================================
# CARREGAR COMPLEX
# ==========================================================

print("Carregando embeddings...")

embedding_matrices, entity2idx, rel2idx, idx2rel = get_embeddings(
    args.kg_model_path,
    args.kg_model_name
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


# ==========================================================
# CARREGAR GRAFO
# ==========================================================

print("Carregando grafo...")

nx_graph = load_graph(
    args.graph_file
)


# ==========================================================
# PROCESSAMENTO
# ==========================================================

all_questions = []

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

        gold_answers = [

            answer.strip()

            for answer in answers.split("|")

            if answer.strip()
        ]

        try:

            sample = generate_paths(
                question,
                tokenizer,
                path_model,
                valid_relations=METAQA_RELATIONS
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
                args.topk
            )

            question_result = {

                "question": question,

                "head_entity": head,

                "gold_answers": gold_answers,

                "candidate_paths": []
            }

            for candidate in candidates:

                question_result[
                    "candidate_paths"
                ].append(

                    {

                        "path":
                            candidate["path"],

                        "score":
                            float(
                                candidate["final_score"]
                            ),

                        "triples":
                            [

                                list(triple)

                                for triple
                                in candidate["triples"]

                            ]
                    }
                )

            all_questions.append(
                question_result
            )

        except Exception as e:

            print(
                f"Erro na pergunta {line_id + 1}:",
                e
            )

        if (line_id + 1) % 100 == 0:

            print(
                f"Processadas {line_id + 1} perguntas"
            )


# ==========================================================
# SALVAR JSON
# ==========================================================

print(
    f"Salvando {len(all_questions)} perguntas..."
)

with open(
    args.output_file,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        all_questions,
        f,
        ensure_ascii=False,
        indent=2
    )

print("Concluído.")