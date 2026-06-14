import pandas as pd

df1 = pd.read_csv("Data/Path_gen/data_train_t5.csv")
df2 = pd.read_csv("Data/QA_data/WQSP/train_wqsp.csv")

df = pd.concat([df1, df2], ignore_index=True)

print(len(df))

df.to_csv(
    "Data/QA_data/WQSP/webqsp_train_full.csv",
    index=False
)