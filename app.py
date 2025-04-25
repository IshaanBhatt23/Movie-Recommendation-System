import pandas as pd
import difflib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
movies = pd.read_csv(r"C:\Users\KIIT\Desktop\Projects\Movie Recommendation\movies.csv")
movies.fillna("", inplace=True)
movies["combined_features"] = (
    movies["genres"] + " " + movies["keywords"] + " " + 
    movies["tagline"] + " " + movies["cast"] + " " + movies["director"]
)
vectorizer = TfidfVectorizer(stop_words="english")
feature_matrix = vectorizer.fit_transform(movies["combined_features"])
cosine_sim = cosine_similarity(feature_matrix, feature_matrix)
def get_closest_match(movie_title):
    all_titles = movies["title"].tolist()
    closest_matches = difflib.get_close_matches(movie_title, all_titles, n=1, cutoff=0.5)
    return closest_matches[0] if closest_matches else None
def recommend_movies(movie_title):
    matched_title = get_closest_match(movie_title)
    if not matched_title:
        return None, ["Movie not found. Try another title!"]
    movie_idx = movies[movies["title"] == matched_title].index[0]
    similarity_scores = list(enumerate(cosine_sim[movie_idx]))
    similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
    recommended_movies = []
    for idx, score in similarity_scores[1:21]:  
        recommended_movies.append(movies.iloc[idx]["title"])
    return matched_title, recommended_movies
    

import streamlit as st
script_path = r"C:\Users\KIIT\Desktop\Projects\Movie Recommendation\movierecommendationsystem.py"
with open(script_path, "r") as file:
    exec(file.read())
st.title("Movie Recommendation System")
st.write("Enter a movie title to get similar movie recommendations.")
movie_title = st.text_input("Enter Movie Title", value=" ")
if st.button("Get Recommendations"):
    matched_title, recommendations = recommend_movies(movie_title)
    if matched_title:
        st.subheader(f"Showing results for: **{matched_title}**")
    st.subheader("Recommended Movies:")
    for movie in recommendations:
        st.write(f"- {movie}")