def generate_paths(question):

    inputs = tokenizer(
        question,
        return_tensors="pt"
    ).to(device)

    outputs = model.generate(
        **inputs,
        max_length=18,
        num_beams=5,
        num_return_sequences=5,
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

    return paths, scores