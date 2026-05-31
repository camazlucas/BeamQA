import os
import torch
import pandas as pd


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