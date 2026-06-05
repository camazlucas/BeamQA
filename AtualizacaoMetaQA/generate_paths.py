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
    num_beams=20,
    num_return_sequences=20,
    max_length=18
):

    valid_relations = {
        "directed_by",
        "has_genre",
        "has_imdb_rating",
        "has_imdb_votes",
        "has_tags",
        "in_language",
        "release_year",
        "starred_actors",
        "written_by",
        "directed_by_inv",
        "has_genre_inv",
        "has_imdb_rating_inv",
        "has_imdb_votes_inv",
        "has_tags_inv",
        "in_language_inv",
        "release_year_inv",
        "starred_actors_inv",
        "written_by_inv"
    }

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

        unique_paths = {}

        for output, score in zip(
            outputs.sequences,
            outputs.sequences_scores
        ):

            path = tokenizer.decode(
                output,
                skip_special_tokens=True
            ).strip()

            relations = path.split()

            if not relations:
                continue

            if not all(
                rel in valid_relations
                for rel in relations
            ):
                continue

            if path not in unique_paths:
                unique_paths[path] = score.item()

        paths = list(unique_paths.keys())

        if len(unique_paths) > 0:

            scores = torch.softmax(
                torch.tensor(
                    list(unique_paths.values())
                ),
                dim=0
            ).tolist()

        else:

            scores = []

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