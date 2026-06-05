import torch

from transformers import (
    BartTokenizer,
    BartForConditionalGeneration
)

MODEL_DIR = "./Modulo1/bart_metaqa_all_hops/final_model/"

device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = BartTokenizer.from_pretrained(MODEL_DIR)

model = BartForConditionalGeneration.from_pretrained(MODEL_DIR)

model.to(device)
model.eval()


def generate_paths(
    input_file,
    num_beams=5,
    num_return_sequences=5,
    max_length=18
):

    with open(input_file, "r", encoding="utf-8") as f:

        questions = [
            line.strip()
            for line in f
            if line.strip()
        ]

    results = []

    for question in questions:

        inputs = tokenizer(
            question,
            return_tensors="pt"
        ).to(device)

        outputs = model.generate(
            **inputs,
            max_length=max_length,
            num_beams=num_beams,
            num_return_sequences=num_return_sequences,
            early_stopping=True,
            output_scores=True,
            return_dict_in_generate=True
        )

        paths = []
        scores = []

        for output, score in zip(
            outputs.sequences,
            outputs.sequences_scores
        ):

            path = tokenizer.decode(
                output,
                skip_special_tokens=True
            )

            paths.append(path)
            scores.append(score.item())

        results.append(
            {
                "question": question,
                "paths": paths,
                "scores": scores
            }
        )

    return results


if __name__ == "__main__":

    input_file = "questions.txt"

    results = generate_paths(input_file)

    for sample in results:

        print("\nPergunta:")
        print(sample["question"])

        print("\nCaminhos:")

        for path, score in zip(
            sample["paths"],
            sample["scores"]
        ):

            print(path)
            print(score)

        print("-" * 80)