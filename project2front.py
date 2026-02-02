import streamlit as st
import pandas as pd
import difflib
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="Movie Recommendation System", page_icon="🎬", layout="centered")

st.title("🎬 Movie Recommendation System")
st.write("Get movie recommendations based on genre/tags similarity 🏷️")

@st.cache_data
def load_data():
    file_path = r"C:\Users\HP\OneDrive\IWT\Desktop\INTERSHIP PROGRAM\movies.csv (1).zip"
    df = pd.read_csv(file_path)
    df = df.fillna("")
    return df

df = load_data()

possible_cols = ["genre", "genres", "overview", "keywords", "cast", "director"]
available_cols = [c for c in possible_cols if c in df.columns]

df["tags"] = ""
for col in available_cols:
    df["tags"] = df["tags"] + " " + df[col].astype(str)

df["tags"] = df["tags"].str.lower()

@st.cache_resource
def train_model(data):
    cv = CountVectorizer(max_features=5000, stop_words="english")
    vectors = cv.fit_transform(data["tags"])
    return cv, vectors

cv, vectors = train_model(df)

df["title"] = df["title"].astype(str)
df["title_lower"] = df["title"].str.lower()

def recommend(movie_name, top_n=5):
    movie_name = movie_name.lower().strip()

    if movie_name not in df["title_lower"].values:
        return []

    index = df[df["title_lower"] == movie_name].index[0]

    sim_scores = cosine_similarity(vectors[index], vectors).flatten()

    movies_list = sorted(list(enumerate(sim_scores)), reverse=True, key=lambda x: x[1])[1:top_n+1]

    recommendations = [df.iloc[i[0]]["title"] for i in movies_list]
    return recommendations

# UI INPUT (Typing)

selected_movie = st.text_input("🎥 Type a movie name")

# Approximate Suggestions

if selected_movie:
    all_titles = df["title"].tolist()

    # Get close matches (approx spelling)
    close_matches = difflib.get_close_matches(
        selected_movie, all_titles, n=5, cutoff=0.4
    )

    if close_matches:
        st.write("Did you mean❓")
        for m in close_matches:
            if st.button(m):   # click to auto-fill
                selected_movie = m
                st.rerun()
    else:
        st.warning("⚠️ No similar movie found, try another spelling.")

# Recommend Button
if st.button("✅ Recommend Movies"):
    results = recommend(selected_movie, top_n=5)

    if results:
        st.subheader("📽️Recommended Movies:")
        for movie in results:
            st.write(f"🎥🎞️{movie}")
    else:
        st.error("❌ Movie not found! Please select from suggestion list.")
