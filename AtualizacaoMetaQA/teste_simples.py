# from generate_paths import generate_paths
# from beamQA import path_finder_candidates
# from utils import get_embeddings, load_graph
# from Model import Model
# from predict_answer import predict_answer

# device = "cuda:0"

# kg_model_path = "../Data/Graph_data/MetaQA/MetaQAProcess/new_complex_metaqa_100_inv/"
# kg_model_name = "trained_model.pkl"

# embedding_matrices, entity2idx, rel2idx, idx2rel = get_embeddings(
#     kg_model_path,
#     kg_model_name
# )

# idx2entity = {v:k for k,v in entity2idx.items()}

# model = Model(
#     embedding_matrices,
#     0.1,
#     True,
#     True
# ).to(device)

# nx_graph = load_graph(
#     "../Data/Graph_data/MetaQA/MetaQAProcess/new_complex_metaqa_100_inv/MetaQA_graph.pkl"
# )


# # carregar modelo e grafo

# results = generate_paths("questions.txt")

# sample = results[0]

# head = "Catch Me If You Can"

# topk = 5

# while True:

#     question = input("\nPergunta (ou 'sair'): ").strip()

#     if question.lower() == "sair":
#         break

#     head = input("Head entity: ").strip()

#     with open(
#         "questions.txt",
#         "w",
#         encoding="utf-8"
#     ) as f:

#         f.write(question + "\n")

#     results = generate_paths("questions.txt")

#     sample = results[0]

#     candidates = path_finder_candidates(
#         head,
#         sample["paths"],
#         sample["scores"],
#         model,
#         entity2idx,
#         idx2entity,
#         rel2idx,
#         nx_graph,
#         device,
#         topk
#     )

#     answer = predict_answer(candidates)

#     print("\nResposta:")
#     print(answer["answer"])

#     print("\nScore:")
#     print(answer["score"])

#     print("\nPath:")
#     print(answer["path"])

#     print("\nTriples:")
#     for triple in answer["triples"]:
#         print(triple)

from generate_paths import (
    generate_paths,
    load_bart_model
)

from generate_paths_rog import (
    generate_paths_rog,
    load_rog_model
)

from utils import get_embeddings

# ==========================================
# MODELOS
# ==========================================

BART_MODEL = "WQSP/Modulo 1/Modelo1-BART/final_model"
ROG_MODEL = "Path_generation/RoG"

KG_MODEL_PATH = "Data/Graph_data/FreeBase/complex_freebase_64_inv"
KG_MODEL_NAME = "trained_model.pkl"

# ==========================================
# RELAÇÕES VÁLIDAS
# ==========================================

_, _, rel2idx, _ = get_embeddings(
    KG_MODEL_PATH,
    KG_MODEL_NAME
)

valid_relations = set(
    rel2idx.keys()
)

print(
    f"Relações válidas: {len(valid_relations)}"
)

# ==========================================
# CARREGAR
# ==========================================

print("Carregando BART...")
bart_tokenizer, bart_model = load_bart_model(
    BART_MODEL
)

print("Carregando RoG...")
rog_tokenizer, rog_model = load_rog_model(
    ROG_MODEL
)

# ==========================================
# PERGUNTAS
# ==========================================

questions = [

    "what does jamaican people speak [m.03_r3]",

    "who plays ken barlow in coronation street [m.01_2n]",

    "where is jamarcus russell from [m.0cjcdj]",

    "where was george washington carver from [m.03djm]",

    "who directed catch me if you can [m.0f4xvl]"
]

# ==========================================
# TESTE
# ==========================================

for question in questions:

    print("\n" + "=" * 80)
    print("PERGUNTA:")
    print(question)

    print("\n----- BART -----")

    bart_result = generate_paths(
        question,
        bart_tokenizer,
        bart_model
    )

    for path, score in zip(
        bart_result["paths"][:10],
        bart_result["scores"][:10]
    ):
        print(f"{score:.4f} | {path}")

    print("\n----- ROG -----")

    rog_result = generate_paths_rog(
        question,
        rog_tokenizer,
        rog_model,
        valid_relations=valid_relations
    )

    print(
        f"Caminhos válidos: {len(rog_result['paths'])}"
    )

    for path, score in zip(
        rog_result["paths"][:10],
        rog_result["scores"][:10]
    ):
        print(f"{score:.4f} | {path}")

    print("=" * 80)