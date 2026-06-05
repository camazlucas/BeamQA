while True:

    question = input("Pergunta: ")

    t0 = time.perf_counter()

    head = extract_entity(question)

    paths, scores = bart_generate(question)

    t1 = time.perf_counter()

    answer, score = path_finder_rec(
        head,
        paths,
        scores,
        model,
        entity2idx,
        idx2entity,
        rel2idx,
        nx_graph,
        device,
        topk=5
    )

    t2 = time.perf_counter()

    print("Resposta:", answer)
    print("Tempo BART:", t1 - t0)
    print("Tempo BeamQA:", t2 - t1)
    print("Tempo Total:", t2 - t0)