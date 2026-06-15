import torch

from transformers import (
    BartTokenizer,
    BartForConditionalGeneration
)

MODEL_DIR = "./Modelo1-BART/final_model"

tokenizer = BartTokenizer.from_pretrained(MODEL_DIR)

model = BartForConditionalGeneration.from_pretrained(MODEL_DIR)

device = "cuda" if torch.cuda.is_available() else "cpu"

model.to(device)
model.eval()

# def predict(question):
#     inputs = tokenizer(question, return_tensors="pt").to(model.device)
#     outputs = model.generate(**inputs, max_length=128)
#     return tokenizer.decode(outputs[0], skip_special_tokens=True)

def predict(question):

    inputs = tokenizer(
        question,
        return_tensors="pt"
    ).to(model.device)

    outputs = model.generate(
        **inputs,
        max_length=128,
        num_beams=10,
        num_return_sequences=10,
        return_dict_in_generate=True,
        output_scores=True
    )

    for i, (seq, score) in enumerate(
        zip(
            outputs.sequences,
            outputs.sequences_scores
        ),
        start=1
    ):

        path = tokenizer.decode(
            seq,
            skip_special_tokens=True
        )

        print(
            f"{i:02d} | score={score.item():.4f} | {path}"
        )

while True:

    question = input("\nPergunta: ")

    if question.lower() == "exit":
        break

    predict(question)
    # print(predict(question))


# print(predict("who directed movies written by Quentin Tarantino"))