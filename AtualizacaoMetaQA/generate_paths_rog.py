import torch

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM
)

from transformers import logging

logging.set_verbosity_error()

device = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ==========================================================
# LOAD MODEL
# ==========================================================

def load_rog_model(model_name):

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )

    model.eval()

    return tokenizer, model


# ==========================================================
# GENERATE PATHS
# ==========================================================

def generate_paths_rog(
    question,
    tokenizer,
    model,
    num_beams=20,
    num_return_sequences=20,
    max_new_tokens=48
):

    prompt = (
        "Please generate a valid relation path "
        "that can be helpful for answering "
        "the following question:\n\n"
        f"{question}"
    )

    inputs = tokenizer(
        prompt,
        return_tensors="pt"
    ).to(model.device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        num_beams=num_beams,
        num_return_sequences=num_return_sequences,
        output_scores=True,
        return_dict_in_generate=True
    )

    unique_paths = {}

    for output, score in zip(
        outputs.sequences,
        outputs.sequences_scores
    ):

        input_len = inputs["input_ids"].shape[1]

        generated_tokens = output[input_len:]

        path = tokenizer.decode(
            generated_tokens,
            skip_special_tokens=True
        ).strip()

        path = path.replace("</PATH>", "<SEP>")
        path = path.strip()

        if "<SEP>" not in path:
            continue

        relations = [
            rel.strip()
            for rel in path.split("<SEP>")
            if rel.strip()
        ]

        if len(relations) > 2:
            relations = relations[:2]

        path = " ".join(relations)

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
# TESTE LOCAL
# ==========================================================

if __name__ == "__main__":

    MODEL_NAME = "rmanluo/RoG"

    tokenizer, model = load_rog_model(
        MODEL_NAME
    )

    question = (
        "the films that share directors "
        "with the film Catch Me If You Can "
        "were in which languages?"
    )


    sample = generate_paths(
        question,
        tokenizer,
        model
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
        print()