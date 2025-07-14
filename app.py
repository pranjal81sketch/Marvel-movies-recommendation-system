import streamlit as st
import pandas as pd
import difflib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
import base64

# ----- Page Configuration -----
st.set_page_config(page_title="Marvel Movie Recommender", layout="centered")

# ----- Optional Background Image -----
def add_bg_from_local(image_file):
    with open(image_file, "rb") as img:
        encoded = base64.b64encode(img.read()).decode()
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/jpg;base64,{encoded}");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

add_bg_from_local("p1.jpg")

# ----- Custom CSS Styling -----
st.markdown("""
    <style>
        .main-title {
            color: yellow;
            font-size: 30px;
            font-weight: bold;
            text-align: center;
            margin: 10px 0;
            white-space: nowrap;
            
        }
        .subtitle {
            color: white;
            font-size: 22px;
            text-align: center;
            margin-bottom: 30px;
        }
        .stTextInput > div > label {
            font-size: 22px;
            color: white;
        }
        .stTextInput > div > div > input {
            width: 300px;
            font-size: 18px;
        }
        .recommendation {
            font-size: 22px;
            color: black;
            margin-top: 10px;
        }
        .stSlider label {
            color: white;
            font-size: 18px;
        }
    </style>
""", unsafe_allow_html=True)

# ----- Load Data -----
movies_data = pd.read_csv("movie_dataset.csv")
movies_data.fillna('', inplace=True)

# Ensure rating column is numeric
movies_data['vote_average'] = pd.to_numeric(movies_data['vote_average'], errors='coerce').fillna(0)

# ----- Feature Engineering -----
selected_features = ['genres', 'keywords', 'tagline', 'cast', 'director']
movies_data['combined_features'] = movies_data[selected_features].apply(lambda row: ' '.join(row.values.astype(str)), axis=1)

# Vectorization
vectorizer = TfidfVectorizer(stop_words='english')
feature_vectors = vectorizer.fit_transform(movies_data['combined_features'])

# Model
nn_model = NearestNeighbors(metric='cosine', algorithm='brute')
nn_model.fit(feature_vectors)

# ----- UI -----
st.markdown('<div class="main-title">🎬 Marvel Movie Recommendation System</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Enter your favorite Marvel movie below and get top recommendations</div>', unsafe_allow_html=True)

# Input
movie_name = st.text_input("Movie Title:")
min_rating = st.slider("Minimum IMDb Rating", 0.0, 10.0, 5.0, step=0.1)

# ----- Recommendation Logic -----
if movie_name:
    list_of_all_titles = movies_data['title'].tolist()
    close_match = difflib.get_close_matches(movie_name, list_of_all_titles, n=1)

    if close_match:
        matched_title = close_match[0]
        movie_index = movies_data[movies_data.title == matched_title].index[0]

        movie_vector = feature_vectors[movie_index]
        distances, indices = nn_model.kneighbors(movie_vector, n_neighbors=30)

        st.subheader(f"Top Recommendations based on '{matched_title}':")
        count = 0
        for idx in indices[0][1:]:
            recommended_movie = movies_data.iloc[idx]
            rating = recommended_movie['vote_average']

            if rating >= min_rating:
                count += 1
                st.markdown(f"<div class='recommendation'>{count}. {recommended_movie['title']} &nbsp;&nbsp;<small>(IMDb: {rating})</small></div>", unsafe_allow_html=True)

        if count == 0:
            st.warning(f"No movies found with IMDb rating ≥ {min_rating}")
    else:
        st.error("No close match found. Try another movie name.")
