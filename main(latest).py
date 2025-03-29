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
    content_features = content_model['content_features']
    similarity_matrix = content_model['similarity_matrix']
    
    # Apply initial filters
    filtered_content = content_features.copy()
    
    if content_type and content_type != "All":
        content_type = content_type.upper()
        filtered_content = filtered_content[filtered_content['type'] == content_type]
    
    if genre_filter:
        filtered_content = filtered_content[filtered_content['genres'].str.contains(genre_filter, case=False, na=False)]
    
    if year_filter:
        filtered_content = filtered_content[filtered_content['release_year'] == year_filter]
    
    if filtered_content.empty:
        return []
    
    # We need to use the filtered content to find most similar items using the similarity matrix
    movie_ids = filtered_content['id'].tolist()
    
    # For the seed movies, get the top similar movies from the similarity matrix
    recommendations = []
    movie_indices = content_features[content_features['id'].isin(movie_ids)].index.tolist()
    
    # Get all unique movie indices from the similarity matrix
    all_indices = set(similarity_matrix.index)
    
    # Only proceed with indices that are actually in the similarity matrix
    valid_indices = [idx for idx in movie_indices if idx in all_indices]
    
    if not valid_indices:
        return []
    
    # For each valid index, get similar movies
    all_similar_items = set()
    for idx in valid_indices:
        # Get top 5 similar movies for each seed movie
        similar_indices = similarity_matrix.loc[idx].sort_values(ascending=False).index[1:6]
        all_similar_items.update(similar_indices)
    
    # Remove the seed movies from recommendations
    all_similar_items = all_similar_items - set(valid_indices)
    
    # Get the titles and ids for the similar movies
    for idx in all_similar_items:
        # Make sure the index exists in content_features
        if idx not in content_features.index:
            continue
            
        movie_id = content_features.loc[idx, 'id']
        movie_info = titles[titles['id'] == movie_id]
        
        if movie_info.empty:
            continue
            
        movie_title = movie_info['title'].iloc[0]
        
        # Get average rating if available
        avg_rating = user_interactions[user_interactions['id'] == movie_id]['rating'].mean() \
                     if not user_interactions[user_interactions['id'] == movie_id].empty else 3.0
                     
        recommendations.append({
            "title": movie_title,
            "rating": float(avg_rating),
            "id": movie_id
        })
    
    # Sort by rating and return top 10
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
