import torch

from transformers import (
    BartTokenizer,
    BartForConditionalGeneration
)

MODEL_DIR = "./bart_metaqa_all_hops/final_model"

tokenizer = BartTokenizer.from_pretrained(MODEL_DIR)

model = BartForConditionalGeneration.from_pretrained(MODEL_DIR)

device = "cuda" if torch.cuda.is_available() else "cpu"

model.to(device)
model.eval()

while True:

    question = input("\nPergunta: ")

    if question.lower() == "exit":
        break

    inputs = tokenizer(
        question,
        return_tensors="pt"
    ).to(device)

    outputs = model.generate(
        **inputs,
        max_length=8,
        num_beams=5,
        num_return_sequences=5,
        early_stopping=True
    )

    print("\nCaminhos gerados:")

    for i, output in enumerate(outputs):

        path = tokenizer.decode(
            output,
            skip_special_tokens=True
        )

        print(f"{i+1}: {path}")