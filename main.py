import streamlit as st
import joblib
from joblib import load
import pandas as pd
import numpy as np
import os
from scipy.sparse import csr_matrix, hstack
from sklearn.metrics.pairwise import cosine_similarity
from ast import literal_eval

# Page configuration
st.set_page_config(
    page_title="Movie Recommender System",
    page_icon="🎬",
    layout="wide"
)

# Apply custom CSS
st.markdown("""
    <style>
    .movie-card {
        background-color: #3E3D53;
        border-radius: 10px;
        padding: 15px;
        margin-bottom: 15px;
    }
    .movie-title {
        font-size: 20px;
        font-weight: bold;
        color: #63C5DA;
    }
    .movie-details {
        font-size: 14px;
        color: #E89149;
    }
    .movie-description {
        font-size: 14px;
        margin-top: 10px;
    }
    .movie-score {
        font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True)

# Check if necessary files exist
required_files = ['collaborative_model.joblib', 'content_model.joblib', 'hybrid_model.joblib', 'titles.xls', 'user_interactions.xls']
missing_files = [file for file in required_files if not os.path.exists(file)]

if missing_files:
    st.error(f"Missing files: {', '.join(missing_files)}")
    st.info("Please make sure all required model and data files are in the same directory as this script.")
    st.stop()

# Load models and data
try:
    with st.spinner("Loading models and data..."):
        collaborative_model = load('collaborative_model.joblib')
        content_model = load('content_model.joblib')
        hybrid_model = load('hybrid_model.joblib')
        titles = pd.read_csv('titles.xls')
        user_interactions = pd.read_csv('user_interactions.xls')
        
        # Handle string lists in dataframe
        for col in ['genres', 'production_countries']:
            if col in titles.columns:
                titles[col] = titles[col].apply(lambda x: literal_eval(x) if isinstance(x, str) else [])
except Exception as e:
    st.error(f"Error loading files: {e}")
    st.stop()

# Sidebar for navigation
st.sidebar.title("🎬 Movie Recommender")
st.sidebar.markdown("---")

# Main content
st.title("Netflix TV Shows & Movie Recommendation System")
st.markdown("Get personalized movie/show recommendations based on different recommendation algorithms.")

recommender_type = st.sidebar.selectbox(
    "Choose Recommendation Method",
    ["Collaborative Filtering", "Content-Based", "Hybrid"],
    index=2  # Default to Hybrid
)

st.sidebar.markdown("---")

# User input section - MODIFIED VERSION
with st.sidebar:
    if recommender_type in ["Collaborative Filtering", "Hybrid"]:
        # Get list of unique user IDs from interactions
        unique_users = sorted(user_interactions['user_id'].unique().tolist()[:100])  # Limit to first 100 for UI performance
        user_id = st.selectbox("Select User ID", unique_users, index=0)
    
    # Only show content filters for Content-Based and Hybrid
    if recommender_type in ["Content-Based", "Hybrid"]:
        st.subheader("Content Filters")
        content_type = st.radio("Movie Type", ["All", "Movie", "Show"])
        
        # Extract all genres and sort them
        all_genres = set()
        for genres_list in titles['genres']:
            if isinstance(genres_list, list):
                all_genres.update(genres_list)
        genre_options = sorted(list(all_genres))
        
        genre_filter = st.selectbox("Filter by Genre", ["All"] + genre_options)
        if genre_filter == "All":
            genre_filter = None
            
        year_range = titles['release_year'].dropna().astype(int)
        min_year, max_year = int(year_range.min()), int(year_range.max())
        
        use_year_filter = st.checkbox("Filter by Release Year")
        if use_year_filter:
            year_filter = st.slider("Select Release Year", min_year, max_year, 2015)
        else:
            year_filter = None
    else:
        # For Collaborative Filtering, set defaults
        content_type = "All"
        genre_filter = None
        year_filter = None


# Function to get movie details
def get_movie_details(movie_id):
    """Get all details for a movie by ID"""
    movie_data = titles[titles['id'] == movie_id]
    if movie_data.empty:
        return {
            'title': 'Unknown',
            'release_year': 'Unknown',
            'genres': [],
            'description': 'No description available',
            'production_countries': [],
            'type': 'Unknown'
        }
    
    movie = movie_data.iloc[0]
    return {
        'title': movie['title'],
        'release_year': movie['release_year'],
        'genres': movie['genres'],
        'description': movie['description'],
        'production_countries': movie['production_countries'],
        'type': movie['type']
    }

# Recommendation Functions
def collaborative_recommendation(user_id, content_type=None):
    """Generate recommendations using collaborative filtering model (SVD)"""
    # Get user's rated movies
    user_rated_movies = user_interactions[user_interactions['user_id'] == user_id]['id'].tolist()
    
    # Get movies not rated by the user
    unrated_movies = titles[~titles['id'].isin(user_rated_movies)].copy()
    
    if content_type and content_type != "All":
        unrated_movies = unrated_movies[unrated_movies['type'] == content_type]
    
    predictions = []
    for _, row in unrated_movies.iterrows():
        try:
            # Predict rating
            est_rating = collaborative_model.predict(user_id, row['id']).est
            
            # Get movie details
            movie_details = get_movie_details(row['id'])
            
            # Add to predictions with complete details
            predictions.append({
                "id": row['id'],
                "title": movie_details['title'],
                "release_year": movie_details['release_year'],
                "genres": movie_details['genres'],
                "description": movie_details['description'],
                "production_countries": movie_details['production_countries'],
                "type": movie_details['type'],
                "rating": est_rating
            })
        except Exception as e:
            continue
    
    return sorted(predictions, key=lambda x: x['rating'], reverse=True)[:10]

def content_based_recommendation(genre_filter=None, year_filter=None, content_type=None):
    """Generate recommendations using content-based filtering"""
    # Extract components from the content model
    content_features = content_model['content_features']
    similarity_matrix = content_model['similarity_matrix']
    
    # Apply filters
    filtered_content = content_features.copy()
    
    if content_type and content_type != "All":
        filtered_content = filtered_content[filtered_content['type'] == content_type]
    
    if genre_filter:
        # Filter by genre - this needs to handle list format
        filtered_content = filtered_content[filtered_content['genres'].apply(
            lambda x: genre_filter in x if isinstance(x, list) else False
        )]
    
    if year_filter:
        filtered_content = filtered_content[filtered_content['release_year'] == year_filter]
    
    if filtered_content.empty:
        return []
    
    # Get movie indices for filtered content
    movie_indices = filtered_content.index.tolist()
    
    # Get all valid indices that exist in the similarity matrix
    valid_indices = [idx for idx in movie_indices if idx in similarity_matrix.index]
    
    if not valid_indices:
        return []
    
    # For each seed movie, get similar movies
    all_similar_items = set()
    for idx in valid_indices[:5]:  # Limit to 5 seed movies for efficiency
        similar_indices = similarity_matrix.loc[idx].sort_values(ascending=False).index[1:6]
        all_similar_items.update(similar_indices)
    
    # Remove the seed movies
    all_similar_items = all_similar_items - set(valid_indices)
    
    # Get complete details for recommendations
    recommendations = []
    for idx in all_similar_items:
        try:
            movie_id = content_features.loc[idx, 'id']
            
            # Get average rating
            avg_rating = user_interactions[user_interactions['id'] == movie_id]['rating'].mean()
            if np.isnan(avg_rating):
                avg_rating = 3.0  # Default rating if no data
                
            # Get movie details
            movie_details = get_movie_details(movie_id)
            
            recommendations.append({
                "id": movie_id,
                "title": movie_details['title'],
                "release_year": movie_details['release_year'],
                "genres": movie_details['genres'],
                "description": movie_details['description'],
                "production_countries": movie_details['production_countries'],
                "type": movie_details['type'],
                "rating": float(avg_rating)
            })
        except Exception as e:
            continue
    
    return sorted(recommendations, key=lambda x: x['rating'], reverse=True)[:10]

def hybrid_recommendation(user_id, genre_filter=None, year_filter=None, content_type=None):
    """Generate recommendations using hybrid approach (collaborative + content-based)"""
    # Extract weights from hybrid model
    collaborative_weight = hybrid_model['genre_weights'][0]
    content_weight = hybrid_model['genre_weights'][1]
    
    # Get collaborative filtering recommendations
    collab_recs = collaborative_recommendation(user_id, content_type)
    
    # Get content-based recommendations
    content_recs = content_based_recommendation(genre_filter, year_filter, content_type)
    
    # Combine recommendations with weights
    movie_scores = {}
    
    # Process collaborative recommendations
    for rec in collab_recs:
        movie_scores[rec['id']] = {
            'score': rec['rating'] * collaborative_weight,
            'cf_score': rec['rating'],
            'cbf_score': 0,
            'details': rec
        }
    
    # Process content recommendations
    for rec in content_recs:
        if rec['id'] in movie_scores:
            movie_scores[rec['id']]['score'] += rec['rating'] * content_weight
            movie_scores[rec['id']]['cbf_score'] = rec['rating']
        else:
            movie_scores[rec['id']] = {
                'score': rec['rating'] * content_weight,
                'cf_score': 0,
                'cbf_score': rec['rating'],
                'details': rec
            }
    
    # Convert to list format with all details
    hybrid_recs = []
    for movie_id, data in movie_scores.items():
        movie_data = data['details']
        movie_data['hybrid_score'] = data['score']
        movie_data['cf_score'] = data['cf_score']
        movie_data['cbf_score'] = data['cbf_score']
        hybrid_recs.append(movie_data)
    
    # Sort by hybrid score and return top 10
    return sorted(hybrid_recs, key=lambda x: x['hybrid_score'], reverse=True)[:10]

# Display movie information
def display_movie_card(movie, recommender_type):
    # Format genres as string
    if isinstance(movie.get('genres'), list):
        genres = ", ".join(movie['genres'])
    else:
        genres = movie.get('genres', 'Not specified')
    
    # Format countries as string
    if isinstance(movie.get('production_countries'), list):
        countries = ", ".join(movie['production_countries']) if movie.get('production_countries') else 'Not specified'
    else:
        countries = movie.get('production_countries', 'Not specified')
    
    # Create HTML card
    st.markdown(f"""
    <div class="movie-card">
        <div class="movie-title">{movie['title']} ({movie.get('release_year', 'N/A')})</div>
        <div class="movie-details">
            <strong>Type:</strong> {movie.get('type', 'N/A')} | 
            <strong>Genres:</strong> {genres} | 
            <strong>Countries:</strong> {countries}
        </div>
        <div class="movie-description">{movie.get('description', 'No description available')}</div>
    """, unsafe_allow_html=True)
    
    # Display appropriate score based on recommender type
    if recommender_type == "Collaborative Filtering":
        st.markdown(f"""
        <div class="movie-score">Predicted Rating: {movie.get('rating', 0):.2f} / 5.0</div>
        """, unsafe_allow_html=True)
    elif recommender_type == "Content-Based":
        st.markdown(f"""
        <div class="movie-score">Average Rating: {movie.get('rating', 0):.2f} / 5.0</div>
        """, unsafe_allow_html=True)
    else:  # Hybrid
        st.markdown(f"""
        <div class="movie-score">
            Hybrid Score: {movie.get('hybrid_score', 0):.2f} / 5.0 
            (CF: {movie.get('cf_score', 0):.2f}, CBF: {movie.get('cbf_score', 0):.2f})
        </div>
        """, unsafe_allow_html=True)

# Generate recommendations on button click
if st.button("Get Recommendations"):
    with st.spinner("Generating recommendations..."):
        try:
            if recommender_type == "Collaborative Filtering":
                # Force 'All' type for collaborative filtering
                recommendations = collaborative_recommendation(user_id, content_type="All")
                st.subheader(f"Top Netflix TV Shows & Movie for User {user_id} (Collaborative Filtering)")
            elif recommender_type == "Content-Based":
                recommendations = content_based_recommendation(genre_filter, year_filter, content_type)
                filters_text = []
                if genre_filter:
                    filters_text.append(f"Genre: {genre_filter}")
                if year_filter:
                    filters_text.append(f"Year: {year_filter}")
                if content_type != "All":
                    filters_text.append(f"Type: {content_type}")
                
                filter_display = f" ({', '.join(filters_text)})" if filters_text else ""
                st.subheader(f"Top Netflix TV Shows & Movie Based on Content{filter_display}")
            else:  # Hybrid
                recommendations = hybrid_recommendation(user_id, genre_filter, year_filter, content_type)
                st.subheader(f"Top Netflix TV Shows & Movie for User {user_id} (Hybrid Recommendations)")
            
            if not recommendations:
                st.warning("No recommendations found matching your criteria. Try adjusting your filters.")
            else:
                # Display all recommendations
                for movie in recommendations:
                    display_movie_card(movie, recommender_type)
                
                # Show metrics
                st.sidebar.markdown("---")
                st.sidebar.subheader("Recommendation Metrics")
                if recommender_type == "Hybrid":
                    avg_hybrid = np.mean([r.get('hybrid_score', 0) for r in recommendations])
                    avg_cf = np.mean([r.get('cf_score', 0) for r in recommendations])
                    avg_cbf = np.mean([r.get('cbf_score', 0) for r in recommendations])
                    
                    col1, col2, col3 = st.sidebar.columns(3)
                    col1.metric("Avg Hybrid", f"{avg_hybrid:.2f}")
                    col2.metric("Avg CF", f"{avg_cf:.2f}")
                    col3.metric("Avg CBF", f"{avg_cbf:.2f}")
                else:
                    avg_rating = np.mean([r.get('rating', 0) for r in recommendations])
                    st.sidebar.metric("Average Rating", f"{avg_rating:.2f}/5.0")
                
        except Exception as e:
            st.error(f"Error generating recommendations: {e}")
            st.expander("Show error details").code(str(e))

# Footer
st.markdown("---")
st.markdown("Netflix TV Shows & Movie Recommender System - LGL")
