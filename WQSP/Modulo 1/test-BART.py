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

def predict(question):
    inputs = tokenizer(question, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_length=128)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

while True:

    question = input("\nPergunta: ")

    if question.lower() == "exit":
        break

    print(predict(question))


# print(predict("who directed movies written by Quentin Tarantino"))