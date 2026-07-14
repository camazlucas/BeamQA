import torch

from transformers import (
    BartTokenizer,
    BartForConditionalGeneration
)

device = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

# ==========================================================
# METAQA RELATIONS
# ==========================================================

METAQA_RELATIONS = {

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

# ==========================================================
# LOAD MODEL
# ==========================================================

def load_model(model_dir, model_type):

    if model_type == "bart":

        tokenizer = BartTokenizer.from_pretrained(
            model_dir
        )

        model = BartForConditionalGeneration.from_pretrained(
            model_dir
        )

        model.to(device)

        model.eval()

        return tokenizer, model
    
    else: 

        raise ValueError(
            f"Modelo '{model_type}' não suportado."
        )


# ==========================================================
# IMPLEMENTAÇÕES DOS MODELOS
# ==========================================================

def _generate_bart(
    question,
    tokenizer,
    model,
    valid_relations,
    num_beams,
    num_return_sequences,
    max_length
):

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

        # ----------------------------------
        # FILTRO OPCIONAL DE RELAÇÕES
        # ----------------------------------

        if valid_relations is not None:

            if not all(
                rel in valid_relations
                for rel in relations
            ):
                continue

        if path not in unique_paths:

            unique_paths[path] = score.item()

    paths = list(
        unique_paths.keys()
    )

    if len(unique_paths) > 0:

        scores = torch.softmax(
            torch.tensor(
                list(
                    unique_paths.values()
                )
            ),
            dim=0
        ).tolist()

    else:

        scores = []

    return {

        "question": question,
        "paths": paths,
        "scores": scores
    }

# ==========================================================
# GENERATE PATHS
# ==========================================================

def generate_paths(
    question,
    tokenizer,
    model,
    model_type,
    valid_relations=None,
    num_beams=20,
    num_return_sequences=20,
    max_length=18
):

    if model_type == "bart":

        return _generate_bart(
            question,
            tokenizer,
            model,
            valid_relations=valid_relations,
            num_beams=num_beams,
            num_return_sequences=num_return_sequences,
            max_length=max_length
        )

    else:

        raise ValueError(
            f"Modelo '{model_type}' não suportado."
        )

# ==========================================================
# TESTE LOCAL
# ==========================================================

if __name__ == "__main__":

    model_type = "bart"

    MODEL_DIR = (
        "../../Dataset/LLM_Data/BART_MetaQA/bart_metaqa_all_hops/final_model/"
    )

    tokenizer, model = load_model(
        MODEL_DIR,
        model_type
    )

    question = (
        "the films that share directors "
        "with the film [Catch Me If You Can] "
        "were in which languages?"
    )

    sample = generate_paths(
        question,
        tokenizer,
        model,
        model_type,
        valid_relations=None
    )

    print("\nPergunta:")
    print(sample["question"])

    print("\nCaminhos:")

    for path, score in zip(
        sample["paths"],
        sample["scores"]
    ):

        print(path)
        print(score)