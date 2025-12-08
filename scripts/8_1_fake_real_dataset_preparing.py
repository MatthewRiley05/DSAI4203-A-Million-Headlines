import pandas as pd


# ===============================
# Fake or Real Dataset (~6.3k, 50/50)
# https://www.kaggle.com/datasets/hassanamin/textdb3?resource=download
# ===============================

df_fr = pd.read_csv("dataset/fake_or_real_news.csv")
df_fr = df_fr[["title", "label"]].rename(columns={"label": "real"})

# Convert REAL/FAKE to 1/0
df_fr["real"] = df_fr["real"].map({"REAL": 1, "FAKE": 0})
df_fr = df_fr.dropna(subset=["title"])
df_fr = df_fr.drop_duplicates()



# ===============================
# FakeNewNet Dataset (~23k, 25/75)
# https://www.kaggle.com/datasets/algord/fake-news
# ===============================
df_fnn = pd.read_csv("dataset/FakeNewsNet.csv")

# Remove where tweet_num == 0, and missing URLs
df_fnn = df_fnn[df_fnn["tweet_num"] > 0]
df_fnn = df_fnn.dropna(subset=["news_url"])

df_fnn = df_fnn[["title", "real"]]
df_fnn = df_fnn.dropna(subset=["title"])
df_fnn["real"] = df_fnn["real"].astype(int)
df_fnn = df_fnn.drop_duplicates()



# ===============================
# WELFake Dataset (~72k, almost 50/50 distribution)
# https://www.kaggle.com/datasets/saurabhshahane/fake-news-classification
# ===============================
df_wel = pd.read_csv("dataset/WELFake_Dataset.csv")
df_wel = df_wel[["title", "label"]].rename(columns={"label": "real"})
df_wel = df_wel.dropna(subset=["title"])
df_wel["real"] = df_wel["real"].astype(int)
df_wel = df_wel.drop_duplicates()

# ==============================================================
# Merging all dfs
# ==============================================================
df_all = pd.concat([df_wel, df_fnn, df_fr], ignore_index=True)
# Drop duplicate or empty titles
df_all = df_all.dropna(subset=["title"])
df_all["title"] = df_all["title"].str.strip()
df_all = df_all[df_all["title"] != ""]

df_all = df_all.drop_duplicates(subset=["title"])

# print all dataset info
print("WELFake:", df_wel.shape, df_wel['real'].value_counts())
print("\nFakeNewsNet:", df_fnn.shape, df_fnn['real'].value_counts())
print("\nFake_or_real:", df_fr.shape, df_fr['real'].value_counts())
print("\n\nFinal merged dataset:", df_all.shape)
print("Merged:", df_all.shape, df_all['real'].value_counts())

# Save
df_all.to_csv("dataset/merged_news_labels.csv", index=False)
print("\nSaved merged dataset to merged_news_labels.csv")