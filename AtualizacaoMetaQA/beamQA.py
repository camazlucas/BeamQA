import torch

def check(head, rel,model,rel2idx,entity2idx,idx2entity,nx_graph,device,topk = 20 ,retscore = False):
    '''
    Computes the scores between head entity and relation
    :param head: head entity
    :param rel: relation
    :param model: QA model
    :param rel2idx: relation to index mapping
    :param entity2idx: entity to index mapping
    :param idx2entity: index to entity mapping
    :param nx_graph: KG
    :param device:
    :param topk: number of k elements to keep
    :param retscore: return score
    :return: sorted list of topk candidates and scores
    '''
    if head not in entity2idx or rel not in rel2idx:
        return [(None, None, None)]
    s_id = entity2idx[head]
    rel_id = rel2idx[rel]
    s = torch.Tensor([s_id]).long().to(device)            # subject indexes
    p = torch.Tensor([rel_id]).long().to(device)          # relation indexes
    scores = model.another_forward(s, p)        # scores of all objects for (s,p,?)

    # ############## DEBUG ###############
    # print(
    #     "max score:",
    #     scores.max().item(),
    #     "min score:",
    #     scores.min().item()
    # )
    # ####################################

    edgeidx = torch.tensor([entity2idx[i[1]]
                            for i in nx_graph.out_edges(head, data='data')
                            if i[2] == rel and i[1] in entity2idx]).long().to(device)
    
    # ################ DEBUG ###################
    # print("HEAD:", head)
    # print("REL:", rel)
    # print("N vizinhos:", len(edgeidx))

    # if len(edgeidx) > 0:
    #     print("Primeiros vizinhos:")
    #     for idx in edgeidx[:10]:
    #         print(idx2entity[idx.item()])   
            
    # #########################################

    
    edge_triples = {}

    for src, dst, relation in nx_graph.out_edges(
        head,
        data='data'
    ):

        if relation == rel and dst in entity2idx:

            edge_triples[dst] = (
                src,
                relation,
                dst
            )

    ############# SUBSTITUIÇÃO #############
    # scores.index_fill_(1, edgeidx, 1)
    # scores.index_fill_(1, s, 0)
    # sc, o = torch.topk(scores, topk, largest=True, dim=-1)  # index of highest-scoring objects
    

    ###############################################
    
    if len(edgeidx) == 0:
        return [(None, None, None)]

    scores_masked = torch.full_like(
        scores,
        -1e9
    )

    scores_masked[0, edgeidx] = scores[0, edgeidx]

    k = min(
        topk,
        len(edgeidx)
    )

    sc, o = torch.topk(
        scores_masked,
        k,
        largest=True,
        dim=-1
    )

    ans = [idx2entity[ent] for ent in o.tolist()[0]]

    answr_score = []

    for entity, score in zip(
        ans,
        sc.tolist()[0]
    ):

# ####################### DEBUG ################################
#         print(
#             "Entidade predita:",
#             entity,
#             "Existe no grafo?",
#             entity in edge_triples
#         )
# ###########################################################


        triple = edge_triples.get(
            entity,
            (head, rel, entity)
        )

        answr_score.append(
            (
                entity,
                score,
                triple
            )
        )

    if retscore:
        return sorted(
            answr_score,
            key=lambda item: item[1],
            reverse=True
        )


    # return [k for k, v in sorted(answr_score.items(), key=lambda item: item[1], reverse=True)]
    return [
        entity
        for entity, score, triple
        in sorted(
            answr_score,
            key=lambda x: x[1],
            reverse=True
        )
    ]

# def check_rec(prev_return, rel,head,topK,model,entity2idx,idx2entity,rel2idx,nx_graph,device):

    
#     if rel not in rel2idx or any(headname not in entity2idx
#                                  for headname in list(zip(*prev_return))[0]):
#         return [(None, None, [])]
#     entity_score = []
#     for prev_entity, prev_score, prev_triples in prev_return:
#         for entity, score, triple in check(prev_entity, rel,model,rel2idx,entity2idx,idx2entity,nx_graph,device,topk = topK, retscore = True):
#             if entity !=  head: entity_score.append((entity, score * prev_score, prev_triples + [triple]))

#     return sorted(entity_score, key= lambda x: x[1], reverse=True)[:topK]

def check_rec(
    prev_return,
    rel,
    head,
    topK,
    model,
    entity2idx,
    idx2entity,
    rel2idx,
    nx_graph,
    device
):

    if (
        rel not in rel2idx
        or any(
            headname not in entity2idx
            for headname in list(zip(*prev_return))[0]
        )
    ):
        return []

    entity_score = []

    for prev_entity, prev_score, prev_triples in prev_return:

        results = check(
            prev_entity,
            rel,
            model,
            rel2idx,
            entity2idx,
            idx2entity,
            nx_graph,
            device,
            topk=topK,
            retscore=True
        )

        for entity, score, triple in results:

            if entity is None:
                continue

            if entity != head:

                entity_score.append(
                    (
                        entity,
                        score * prev_score,
                        prev_triples + [triple]
                    )
                )

    return sorted(
        entity_score,
        key=lambda x: x[1],
        reverse=True
    )[:topK]

# def path_finder_rec(headname, chains,scorez,model,entity2idx,idx2entity,rel2idx,nx_graph,device,topk=10):
#     '''
#     :param headname: Head entity
#     :param chains: list of paths
#     :param scorez: list of scores
#     :param model: QA model
#     :param entity2idx: entity to index mapping
#     :param idx2entity: index to entity mapping
#     :param rel2idx: relation to index mapping
#     :param nx_graph: Knowledge graph
#     :param device:
#     :return: the top predicted entity , score
#     '''
#     max_score = 0
#     predicted_entity = ''
#     predicted_path = ''
#     best_triples = []
#     for path,pscore in zip(chains,scorez):
#         path = path.split()
        
#         prev_return = [(headname, 1, [])]
        
        
#         for j , path_i in enumerate(path):
#             prev_return = check_rec(prev_return, path_i, headname, topk, model, entity2idx, idx2entity,
#                                     rel2idx, nx_graph, device)
            
#             if not prev_return:
#                 break

#         if not prev_return:
#             continue

#         entity, score, triples = prev_return[0]

#         if entity and score * pscore > max_score:
#             predicted_entity = entity
#             max_score = score * pscore
#             best_triples = triples

#     return predicted_entity, max_score, best_triples


############### Versao DEBUG ######################
def path_finder_rec(headname, chains,scorez,model,entity2idx,idx2entity,rel2idx,nx_graph,device,topk=10):
    '''
    :param headname: Head entity
    :param chains: list of paths
    :param scorez: list of scores
    :param model: QA model
    :param entity2idx: entity to index mapping
    :param idx2entity: index to entity mapping
    :param rel2idx: relation to index mapping
    :param nx_graph: Knowledge graph
    :param device:
    :return: the top predicted entity , score
    '''
    max_score = 0
    predicted_entity = ''
    best_triples = []
    for path,pscore in zip(chains,scorez):
        path = path.split()
        
        prev_return = [(headname, 1, [])]

        
        
        
        for j, path_i in enumerate(path):

            print(
                f"Hop {j+1}:",
                path_i,
                "Entrada:",
                prev_return
            )

            prev_return = check_rec(
                prev_return,
                path_i,
                headname,
                topk,
                model,
                entity2idx,
                idx2entity,
                rel2idx,
                nx_graph,
                device
            )

        if not prev_return:
            continue

        entity, score, triples = prev_return[0]

        if entity and score * pscore > max_score:
            predicted_entity = entity
            max_score = score * pscore
            best_triples = triples

    return predicted_entity, max_score, best_triples