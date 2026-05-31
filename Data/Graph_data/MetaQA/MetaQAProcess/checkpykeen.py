import pandas as pd

print(
    pd.read_csv(
        "complex_metaqa/training_triples/entity_to_id.tsv.gz",
        sep="\t"
    ).head()
)

print(
    pd.read_csv(
        "complex_metaqa/training_triples/relation_to_id.tsv.gz",
        sep="\t"
    ).head()
)