import argparse
import pandas as pd
import re

# ==========================
# ARGUMENTOS
# ==========================

parser = argparse.ArgumentParser()

parser.add_argument(
    "--input_dir",
    type=str,
    required=True,
    help="Diretório contendo train_1hop.txt, train_2hop.txt e train_3hop.txt"
)

parser.add_argument(
    "--output_file",
    type=str,
    default="train_all_hops.csv",
    help="Arquivo CSV de saída"
)

args = parser.parse_args()

# ==========================
# CARREGA OS DATASETS
# ==========================

train1 = pd.read_csv(
    f"{args.input_dir}/train_1hop.txt",
    sep="\t",
    names=["QA", "ANS", "TAG"]
)

train2 = pd.read_csv(
    f"{args.input_dir}/train_2hop.txt",
    sep="\t",
    names=["QA", "ANS", "TAG"]
)

train3 = pd.read_csv(
    f"{args.input_dir}/train_3hop.txt",
    sep="\t",
    names=["QA", "ANS", "TAG"]
)

# ==========================
# JUNTA TUDO
# ==========================

train = pd.concat(
    [train1, train2, train3],
    ignore_index=True
)

print("Total de exemplos:", len(train))

# ==========================
# PRÉ-PROCESSAMENTO
# ==========================

def preprocess_question(text):
    return re.sub(r'(\[.*?\])+', "NE", text)

def preprocess_path(path):
    return path.replace("|", " ")

train["text"] = train["QA"].apply(preprocess_question)
train["tag"] = train["TAG"].apply(preprocess_path)

# ==========================
# MANTÉM APENAS O NECESSÁRIO
# ==========================

train_final = train[["text", "tag"]]

print(train_final.head())

# ==========================
# SALVA
# ==========================

train_final.to_csv(
    args.output_file,
    index=False
)

print(f"Arquivo salvo: {args.output_file}")

# ==========================
# RELAÇÕES ÚNICAS
# ==========================

relations = set()

for tag in train_final["tag"]:
    for rel in tag.split():
        relations.add(rel)

print("\nNúmero de relações:", len(relations))

print("\nRelações:")

for rel in sorted(relations):
    print(rel)