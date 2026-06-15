import pandas as pd

# =====================================================
# PERGUNTAS SINTÉTICAS
# =====================================================

df1 = pd.read_csv(
    "Data/Path_gen/data_train_t5.csv"
)

df1 = pd.DataFrame({
    "text": df1["target_text"],
    "tag": df1["prop"]
})

# =====================================================
# WEBQSP ORIGINAL
# =====================================================

df2 = pd.read_csv(
    "Data/QA_data/WQSP/train_wqsp.csv"
)

df2 = pd.DataFrame({
    "text": df2["qa"],
    "tag": (
        df2["rel"]
        .astype(str)
        .str.replace("[", "", regex=False)
        .str.replace("]", "", regex=False)
        .str.replace("'", "", regex=False)
        .str.replace("|", " ", regex=False)
    )
})

# =====================================================
# CONCATENA
# =====================================================

df = pd.concat(
    [df1, df2],
    ignore_index=True
)

# remove possíveis linhas vazias

df = df.dropna(
    subset=["text", "tag"]
)

print("Total de exemplos:", len(df))

print(df.head())

# =====================================================
# SALVA
# =====================================================

df.to_csv(
    "Data/QA_data/WQSP/webqsp_train_full.csv",
    index=False
)