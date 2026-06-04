import os
import torch
import pandas as pd
from torch.utils.data import Dataset

class DatasetMetaQA_all_hops(Dataset):
    def __init__(self, data, entity2idx, rel2idx):
        self.data = data
        self.entity2idx = entity2idx
        self.rel2idx = rel2idx

    def __len__(self):
        return len(self.data)

    def toOneHot(self, indices):
        indices = torch.LongTensor(indices)
        vec_len = len(self.entity2idx)
        one_hot = torch.FloatTensor(vec_len)
        one_hot.zero_()
        one_hot.scatter_(0, indices, 1)
        return one_hot

    def __getitem__(self, index):
        data_point = self.data[index]
        head_id = self.entity2idx[data_point[0].strip()]
        path = [self.rel2idx[rel_name] for rel_name in data_point[2]]
        tail_ids = []
        for tail_name in data_point[1]:
            tail_name = tail_name.strip()
            tail_ids.append(self.entity2idx[tail_name])
        tail_onehot = self.toOneHot(tail_ids)
        head_id = torch.tensor(head_id,dtype=torch.long)
        return head_id, tail_onehot, torch.tensor(path)

class PyKEENLoader():

    def __init__(self, embedding_path):
        self.embedding_path = embedding_path

    def load_pykeen_checkpoint(self, model_path):

        model = torch.load(
            os.path.join(self.embedding_path, model_path),
            map_location="cpu",
            weights_only=False
        )

        entity_df = pd.read_csv(
            os.path.join(
                self.embedding_path,
                "training_triples",
                "entity_to_id.tsv.gz"
            ),
            sep="\t"
        )

        relation_df = pd.read_csv(
            os.path.join(
                self.embedding_path,
                "training_triples",
                "relation_to_id.tsv.gz"
            ),
            sep="\t"
        )

        entity2idx = {
            row["label"]: int(row["id"])
            for _, row in entity_df.iterrows()
        }

        rel2idx = {
            row["label"]: int(row["id"])
            for _, row in relation_df.iterrows()
        }

        original_rels = list(rel2idx.keys())
            
        for rel in original_rels:
            rel2idx[rel + "_inv"] = len(rel2idx)

        print("Número de relações:", len(rel2idx))
        print(rel2idx.keys())
        

        # entity_emb = (
        #     model.entity_representations[0]()
        #     .detach()
        #     .cpu()
        # )

        # relation_emb = (
        #     model.relation_representations[0]()
        #     .detach()
        #     .cpu()
        # )

        # embedding_matrix = [
        #     entity_emb[i]
        #     for i in range(entity_emb.shape[0])
        # ]

        # relation_matrix = [
        #     relation_emb[i]
        #     for i in range(relation_emb.shape[0])
        # ]

        entity_emb = (
            model.entity_representations[0]()
            .detach()
            .cpu()
        )

        relation_emb = (
            model.relation_representations[0]()
            .detach()
            .cpu()
        )

        entity_emb = torch.cat(
            [entity_emb.real, entity_emb.imag],
            dim=1
        )

        relation_emb = torch.cat(
            [relation_emb.real, relation_emb.imag],
            dim=1
        )

        print(entity_emb.shape)
        print(relation_emb.shape)
        print(entity_emb.dtype)

        embedding_matrix = [
            entity_emb[i]
            for i in range(entity_emb.shape[0])
        ]

        relation_matrix = [
            relation_emb[i]
            for i in range(relation_emb.shape[0])
        ]

        return (
            entity2idx,
            rel2idx,
            embedding_matrix,
            relation_matrix
        )