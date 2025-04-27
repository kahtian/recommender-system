import streamlit as st
import joblib
from joblib import load
import pandas as pd
import numpy as np
import os
from scipy.sparse import csr_matrix, hstack
from sklearn.metrics.pairwise import cosine_similarity

# Check if necessary files exist
required_files = ['collaborative_model.joblib', 'content_model.joblib', 'hybrid_model.joblib', 'titles.xls', 'user_interactions.xls']
for file in required_files:
    if not os.path.exists(file):
        st.error(f"File '{file}' not found.")
        st.stop()

# Load models and data
try:
    collaborative_model = load('collaborative_model.joblib')
    content_model = load('content_model.joblib')
    hybrid_model = load('hybrid_model.joblib')
    titles = pd.read_csv('titles.xls', usecols=['id', 'title', 'genres', 'production_countries', 'release_year', 'type', 'description'])
    user_interactions = pd.read_csv('user_interactions.xls', usecols=['user_id', 'id', 'rating'])
except Exception as e:
    st.error(f"Error loading files: {e}")
    st.stop()

# Streamlit UI
st.title("Movie Recommendation System")
recommender_type = st.selectbox("Select Recommendation Method", ["Collaborative Filtering", "Content-Based", "Hybrid"])
user_id = None
genre_filter = None
year_filter = None
content_type = None

if recommender_type in ["Collaborative Filtering", "Hybrid"]:
    user_id = st.number_input("Enter your User ID", min_value=1, max_value=10000, value=1)
if recommender_type in ["Content-Based", "Hybrid"]:
    content_type = st.radio("Filter by Type", ["All", "Movie", "Show"])
    genre_options = ["action", "animation", "comedy", "crime", "documentation", "drama", "european", "family", "fantasy", "history", "horror", "music", "reality", "romance", "scifi", "sport", "thriller", "war", "western"]
    genre_filter = st.selectbox("Choose a genre to filter:", genre_options)
    filter_by_year = st.checkbox("Filter by released year")
    min_year, max_year = 1960, 2022
    year_filter = st.slider("Select a release year", min_year, max_year, 2000) if filter_by_year else None

# Recommendation Functions
def collaborative_recommendation(user_id, content_type=None):
    """Generate recommendations using collaborative filtering model (SVD)"""
    # Use the SVD model to predict ratings for all unrated items
    user_rated_movies = user_interactions[user_interactions['user_id'] == user_id]['id'].tolist()
    unrated_movies = titles[~titles['id'].isin(user_rated_movies)].copy()
    
    if content_type and content_type != "All":
        content_type = content_type.strip().lower()
        unrated_movies = unrated_movies[unrated_movies['type'].str.strip().str.lower() == content_type]
    
    predictions = []
    for _, row in unrated_movies.iterrows():
        try:
            est_rating = collaborative_model.predict(user_id, row['id']).est
            predictions.append({"title": row['title'], "rating": est_rating, "id": row['id']})
        except:
            # Skip if prediction fails for this item
            continue
    
    return sorted(predictions, key=lambda x: x['rating'], reverse=True)[:10]

def content_based_recommendation(genre_filter=None, year_filter=None, content_type=None):
    """Generate recommendations using content-based filtering"""
    # Extract components from the content model
    tfidf_vectorizer = content_model['tfidf_vectorizer']
    scaler = content_model['scaler']
    content_features = content_model['content_features']
    
    # Apply filters to the content features
    filtered_content = content_features.copy()
    
    if content_type and content_type != "All":
        filtered_content = filtered_content[filtered_content['type'].str.strip().str.lower() == content_type.lower()]
    
    if genre_filter:
        filtered_content = filtered_content[filtered_content['genres'].str.contains(genre_filter, case=False, na=False)]
    
    if year_filter:
        filtered_content = filtered_content[filtered_content['release_year'] == year_filter]
    
    if filtered_content.empty:
        return []
    
    # Preprocess text features and transform using saved TF-IDF
    filtered_content['text_features'] = (
        filtered_content['genres'].fillna('') + " " +
        filtered_content['production_countries'].fillna('') + " " +
        filtered_content['description'].fillna('')
    )
    tfidf_matrix = tfidf_vectorizer.transform(filtered_content['text_features'])
    
    # Scale numerical features (release_year)
    numerical_features = scaler.transform(filtered_content[['release_year']])
    numerical_features_sparse = csr_matrix(numerical_features)
    
    # Combine features
    combined_features = hstack([tfidf_matrix, numerical_features_sparse])
    
    # Compute similarity with all content
    similarity_matrix = cosine_similarity(combined_features, tfidf_vectorizer.transform(content_features['text_features']))
    
    # Get top similar items
    recommendations = []
    for idx in range(len(filtered_content)):
        sim_scores = list(enumerate(similarity_matrix[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:6]  # Exclude self
        for movie_idx, score in sim_scores:
            movie_id = content_features.iloc[movie_idx]['id']
            avg_rating = user_interactions[user_interactions['id'] == movie_id]['rating'].mean()
            recommendations.append({
                "title": content_features.iloc[movie_idx]['title'],
                "rating": avg_rating if not pd.isna(avg_rating) else 3.0,
                "id": movie_id
            })
    
    return sorted(recommendations, key=lambda x: x['rating'], reverse=True)[:10]

def hybrid_recommendation(user_id, genre_filter=None, year_filter=None, content_type=None):
    """Generate recommendations using hybrid approach (collaborative + content-based)"""
    # Extract components from hybrid model
    weights = hybrid_model['genre_weights']  # (collaborative_weight, content_weight)
    
    # Get collaborative filtering recommendations
    collab_recs = collaborative_recommendation(user_id, content_type)
    
    # Get content-based recommendations
    content_recs = content_based_recommendation(genre_filter, year_filter, content_type)
    
    # Combine recommendations with weighting
    collab_weight = weights[0]
    content_weight = weights[1]
    
    # Create a dictionary to track combined scores
    combined_scores = {}
    
    # Process collaborative recommendations
    for rec in collab_recs:
        combined_scores[rec['title']] = rec['rating'] * collab_weight
    
    # Process content recommendations
    for rec in content_recs:
        if rec['title'] in combined_scores:
            combined_scores[rec['title']] += rec['rating'] * content_weight
        else:
            combined_scores[rec['title']] = rec['rating'] * content_weight
    
    # Convert back to list
    hybrid_recs = [{"title": title, "rating": score} for title, score in combined_scores.items()]
    
    # Sort and return top 10
    return sorted(hybrid_recs, key=lambda x: x['rating'], reverse=True)[:10]

# Generate Recommendations
if st.button("Get Recommendations"):
    try:
        if recommender_type == "Collaborative Filtering":
            recommendations = collaborative_recommendation(user_id, content_type)
        elif recommender_type == "Content-Based":
            recommendations = content_based_recommendation(genre_filter, year_filter, content_type)
        else:  # Hybrid
            recommendations = hybrid_recommendation(user_id, genre_filter, year_filter, content_type)
        
        st.write("### Top Recommendations:")
        if not recommendations:
            st.warning("No recommendations found matching your criteria. Try widening your filters.")
        else:
            for movie in recommendations:
                rating_text = f"{movie['rating']:.2f}" if isinstance(movie.get('rating'), float) else "N/A"
                
                # Differentiate between predicted and average ratings
                if recommender_type == "Collaborative Filtering":
                    st.write(f"**{movie['title']}** (Predicted Rating: {rating_text})")
                elif recommender_type == "Content-Based":
                    st.write(f"**{movie['title']}** (Average Rating: {rating_text})")
                else:  # Hybrid
                    st.write(f"**{movie['title']}** (Combined Score: {rating_text})")
    except Exception as e:
        st.error(f"Error generating recommendations: {e}")
        import traceback
        st.code(traceback.format_exc())
