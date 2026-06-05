import beam_executor
import predict_answer
import generate_paths
import time

question = input("Pergunta: ")

t0 = time.perf_counter()

paths, score_paths = generate_paths(question)

t1 = time.perf_counter()

candidate_paths, scores = beam_executor(model,
                                        entity2idx,
                                        rel2idx,
                                        idx2entity,
                                        nx_graph,
                                        device)

t2 = time.perf_counter()

answer, score = predict_answer(head,
                                candidate_paths,
                                scores,
                                topk=5
                                )

t3 = time.perf_counter()

# print("Resposta:", answer)
print("Tempo BART:", t1 - t0)
print("Tempo BeamQA:", t2 - t1)
print("Tempo Total:", t3 - t0)