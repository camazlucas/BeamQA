import argparse
import torch
import pandas as pd

from datasets import Dataset

from transformers import (
    BartTokenizer,
    BartForConditionalGeneration,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments
)

# =====================================================
# ARGUMENTOS
# =====================================================

parser = argparse.ArgumentParser()

parser.add_argument(
    "--dataset_path",
    type=str,
    required=True,
    help="Caminho para o CSV de treinamento"
)

parser.add_argument(
    "--output_dir",
    type=str,
    default="./bart_metaqa_all_hops",
    help="Diretório para salvar checkpoints e modelo final"
)

parser.add_argument(
    "--epochs",
    type=int,
    default=100
)

parser.add_argument(
    "--batch_size",
    type=int,
    default=32
)

args = parser.parse_args()

# =====================================================
# VERIFICA CUDA
# =====================================================

print("PyTorch:", torch.__version__)
print("CUDA disponível:", torch.cuda.is_available())

if not torch.cuda.is_available():
    raise RuntimeError(
        "CUDA não está disponível."
    )

print("GPU detectada:", torch.cuda.get_device_name(0))

print(
    "VRAM:",
    round(
        torch.cuda.get_device_properties(0).total_memory / 1024**3,
        2
    ),
    "GB"
)

# =====================================================
# CONFIGURAÇÕES
# =====================================================

MODEL_NAME = "facebook/bart-large"

LEARNING_RATE = 5e-5

WEIGHT_DECAY = 0.01

# =====================================================
# CARREGA DADOS
# =====================================================

print("\nCarregando dataset...")

df = pd.read_csv(args.dataset_path)

print("Total de exemplos:", len(df))

# =====================================================
# DATASET HF
# =====================================================

dataset = Dataset.from_pandas(df)

# =====================================================
# TOKENIZER
# =====================================================

print("\nCarregando tokenizer...")

tokenizer = BartTokenizer.from_pretrained(
    MODEL_NAME
)

# =====================================================
# ADICIONA RELAÇÕES AO VOCABULÁRIO
# =====================================================

relations = set()

for tag in df["tag"]:
    for rel in tag.split():
        relations.add(rel)

print("Número de relações:", len(relations))

tokenizer.add_tokens(list(relations))

print("Vocabulário final:", len(tokenizer))

# =====================================================
# TOKENIZAÇÃO
# =====================================================

def preprocess(examples):

    inputs = tokenizer(
        examples["text"],
        truncation=True,
        max_length=128
    )

    targets = tokenizer(
        examples["tag"],
        truncation=True,
        max_length=16
    )

    inputs["labels"] = targets["input_ids"]

    return inputs


print("\nTokenizando dataset...")

dataset = dataset.map(
    preprocess,
    batched=True,
    remove_columns=dataset.column_names
)

# =====================================================
# MODELO
# =====================================================

print("\nCarregando modelo BART...")

model = BartForConditionalGeneration.from_pretrained(
    MODEL_NAME
)

model.resize_token_embeddings(
    len(tokenizer)
)

# =====================================================
# CONGELA ENCODER
# (como no BeamQA para WQSP)
# =====================================================

for param in model.get_encoder().parameters():
    param.requires_grad = False

print("Encoder congelado.")

# =====================================================
# DATA COLLATOR
# =====================================================

data_collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer,
    model=model
)

# =====================================================
# TREINAMENTO
# =====================================================

training_args = TrainingArguments(
    output_dir=args.output_dir,

    num_train_epochs=args.epochs,

    per_device_train_batch_size=args.batch_size,

    learning_rate=LEARNING_RATE,

    weight_decay=WEIGHT_DECAY,

    save_strategy="epoch",

    save_total_limit=100,

    logging_steps=50,

    report_to="none",

    dataloader_num_workers=4,

    fp16=False
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    data_collator=data_collator,
    processing_class=tokenizer
)

# =====================================================
# TREINAMENTO
# =====================================================

print("\nIniciando treinamento...")

trainer.train()

# =====================================================
# SALVA MODELO FINAL
# =====================================================

final_model_dir = f"{args.output_dir}/final_model"

trainer.save_model(
    final_model_dir
)

tokenizer.save_pretrained(
    final_model_dir
)

print("\nTreinamento concluído.")
print("Modelo salvo em:", final_model_dir)