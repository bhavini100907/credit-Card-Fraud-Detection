import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ✅ 1) LOAD DATA (CSV)
file_path = r"C:\users\hp\OneDrive\IWT\Desktop\INTERSHIP PROGRAM\movies.csv (1).zip"
df = pd.read_csv(file_path)

print("Dataset Shape:", df.shape)
print("Columns:", df.columns)

# ✅ 2) CLEAN DATA
df = df.fillna("")

possible_cols = ["genre", "genres", "overview", "keywords", "cast", "director"]
available_cols = [c for c in possible_cols if c in df.columns]

if "title" not in df.columns:
    raise ValueError("❌ Your CSV must contain a column named 'title'")

if len(available_cols) == 0:
    raise ValueError("❌ Your CSV must contain at least one column like genre/overview/keywords/cast/director")

# Create tags column
df["tags"] = ""
for col in available_cols:
    df["tags"] = df["tags"] + " " + df[col].astype(str)

df["tags"] = df["tags"].str.lower()

print("\n✅ Cleaned Data Preview:")
print(df[["title", "tags"]].head())

# ✅ 3) SPLIT DATA
train_data, test_data = train_test_split(df, test_size=0.2, random_state=42)
train_data = train_data.reset_index(drop=True)
test_data = test_data.reset_index(drop=True)

print("\nTrain Size:", train_data.shape)
print("Test Size:", test_data.shape)


from sklearn.metrics.pairwise import cosine_similarity

cv = CountVectorizer(max_features=5000, stop_words="english")
train_vectors = cv.fit_transform(train_data["tags"])   # keep sparse (DON'T use .toarray())

train_data["title_lower"] = train_data["title"].str.lower()

def recommend(movie_name, top_n=5):
    movie_name = movie_name.lower()

    if movie_name not in train_data["title_lower"].values:
        print("❌ Movie not found! Please type correct name.")
        return

    index = train_data[train_data["title_lower"] == movie_name].index[0]

    # ✅ compute similarity of ONE movie with ALL movies (not full matrix)
    sim_scores = cosine_similarity(train_vectors[index], train_vectors).flatten()

    movies_list = sorted(list(enumerate(sim_scores)), reverse=True, key=lambda x: x[1])[1:top_n+1]

    print(f"\n🎬 Recommended Movies for '{movie_name.title()}':")
    for i in movies_list:
        print(train_data.iloc[i[0]]["title"])


