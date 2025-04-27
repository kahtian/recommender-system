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

# Modify your data loading section
try:
    with st.spinner("Loading models and data..."):
        collaborative_model = load('collaborative_model.joblib')
        content_model = load('content_model.joblib')
        hybrid_model = load('hybrid_model.joblib')
        titles = pd.read_csv('titles.xls')
        user_interactions = pd.read_csv('user_interactions.xls')
        
        # Ensure release_year is numeric in titles dataframe
        titles['release_year'] = pd.to_numeric(titles['release_year'], errors='coerce')
        
        # Handle string lists in dataframe
        for col in ['genres', 'production_countries']:
            if col in titles.columns:
                titles[col] = titles[col].apply(lambda x: literal_eval(x) if isinstance(x, str) else [])
        
        # Ensure content model has all required fields
        if 'content_features' in content_model:
            # Check if release_year exists
            if 'release_year' not in content_model['content_features'].columns:
                # Add release_year from titles if missing
                st.sidebar.write("Adding release_year to content_features")
                content_model['content_features'] = content_model['content_features'].merge(
                    titles[['id', 'release_year']], on='id', how='left')
            
            # Convert release_year to numeric
            content_model['content_features']['release_year'] = pd.to_numeric(
                content_model['content_features']['release_year'], errors='coerce')
            
except Exception as e:
    st.error(f"Error loading files: {e}")
    st.stop()

# Main content
st.title("Netflix TV Shows & Movie Recommendation System")
st.markdown("Get personalized movie/show recommendations based on different recommendation algorithms.")

recommender_type = st.sidebar.selectbox(
    "Choose Recommendation Method",
    ["Collaborative Filtering", "Content-Based", "Hybrid"],
    index=2  # Default to Hybrid
)

st.sidebar.markdown("---")

# User input section
with st.sidebar:
    # Common inputs
    if recommender_type in ["Collaborative Filtering", "Hybrid"]:
        # Get list of unique user IDs from interactions
        unique_users = sorted(user_interactions['user_id'].unique().tolist()[:100])  # Limit to first 100 for UI performance
        user_id = st.selectbox("Select User ID", unique_users, index=0)
    
    # Content filters (for Content-Based and Hybrid)
    if recommender_type in ["Content-Based", "Hybrid"]:
        st.subheader("Content Filters")
        
        # Movie Type (single select)
        content_type = st.radio("Movie Type", ["All", "Movie🎬", "Show 📺"], index=0)
        
        # Genre (multiselect)
        all_genres = set()
        for genres_list in titles['genres']:
            if isinstance(genres_list, list):
                all_genres.update(genres_list)
        genre_options = sorted(list(all_genres))
        selected_genres = st.multiselect("Filter by Genre", genre_options, default=None)
        genre_filter = selected_genres if selected_genres else None
        
        # Release Year (range slider)
        year_range = titles['release_year'].dropna().astype(int)
        min_year, max_year = int(year_range.min()), int(year_range.max())
        year_filter = st.slider(
            "Release Year Range",
            min_value=1945,
            max_value=2022,
            value=(1945, 2022)
        )
        
        # Production Country (single select)
        all_countries = set()
        for countries_list in titles['production_countries']:
            if isinstance(countries_list, list):
                all_countries.update(countries_list)
        country_options = sorted(list(all_countries))
        country_filter = st.selectbox("Production Country", ["All"] + country_options, index=0)
        if country_filter == "All":
            country_filter = None
    else:
        # For Collaborative Filtering, set defaults
        content_type = "All"
        genre_filter = None
        year_filter = (1945, 2022)  # Full range
        country_filter = None

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
def collaborative_recommendation(user_id, content_type="All"):
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


def content_based_recommendation(genre_filter=None, year_filter=None, content_type=None, country_filter=None, n=10):
    content_features_full = content_model['content_features'].copy()

    def safe_literal_eval(x):
        if isinstance(x, str):
            try:
                if x.strip().startswith(('[', '(')) and x.strip().endswith((']', ')')):
                    return literal_eval(x)
                else: return []
            except: return []
        elif isinstance(x, list): return x
        else: return []

    for col in ['genres', 'production_countries']:
        if col in content_features_full.columns:
            try:
                content_features_full[col] = content_features_full[col].apply(safe_literal_eval)
            except Exception as e:
                st.sidebar.error(f"Error applying literal_eval to {col}: {e}")
                return []

    content_features = content_features_full.copy()

    if content_type and content_type != "All":
        if 'type' in content_features.columns:
            content_features = content_features.dropna(subset=['type'])
            content_features['type'] = content_features['type'].str.lower()
            content_features = content_features[content_features['type'] == content_type.lower()]
            
    def match_genres(movie_genres, filter_genres):
        if not isinstance(movie_genres, list): return False
        movie_genres_lower = {str(g).lower() for g in movie_genres}
        filter_genres_lower = {str(g).lower() for g in filter_genres}
        return not movie_genres_lower.isdisjoint(filter_genres_lower)

    if genre_filter:
        content_features = content_features[content_features['genres'].apply(lambda x: match_genres(x, genre_filter))]

    if year_filter:
        if 'release_year' in content_features.columns:
            content_features['release_year'] = pd.to_numeric(content_features['release_year'], errors='coerce')
            content_features = content_features.dropna(subset=['release_year'])
            content_features = content_features[
                (content_features['release_year'] >= year_filter[0]) &
                (content_features['release_year'] <= year_filter[1])
            ]
        else: st.sidebar.warning("'release_year' column not found for filtering.")

    if country_filter:
        if 'production_countries' in content_features.columns:
            content_features = content_features[content_features['production_countries'].apply(
                lambda x: country_filter.lower() in [c.lower() for c in x] if isinstance(x, list) else False
            )]
        else: st.sidebar.warning("'production_countries' column not found for filtering.")


    recommendations = [] # Initialize recommendations list

    if content_features.empty:
        st.sidebar.warning("No items match your filters! Falling back to popular items.")
        # Fallback to popular items
        avg_ratings = user_interactions.groupby('id')['rating'].mean().reset_index()
        fallback_content = content_features_full.merge(avg_ratings, on='id', how='left')
        fallback_content['rating'] = fallback_content['rating'].fillna(3.0) # Use fillna on the merged series
        fallback_content = fallback_content.sort_values('rating', ascending=False).head(n)

        # Corrected fallback loop
        for _, row in fallback_content.iterrows(): # Corrected variable name
            details = get_movie_details(row['id'])
            # --- ADD ID HERE ---
            details['id'] = row['id']
            # --- END ADDITION ---
            details['rating'] = row['rating']
            recommendations.append(details)

    else:
        # Merge with average ratings to rank the filtered items
        avg_ratings = user_interactions.groupby('id')['rating'].mean().reset_index()
        filtered_with_ratings = content_features.merge(avg_ratings, on='id', how='left')
        filtered_with_ratings['rating'] = filtered_with_ratings['rating'].fillna(3.0) # Default rating if no interaction
        top_filtered = filtered_with_ratings.sort_values('rating', ascending=False).head(n)

    return recommendations # Return the list
    

def hybrid_recommendation(user_id, genre_filter=None, year_filter=None, content_type=None, country_filter=None):
    
    # Get recommendations from both models
    collab_recs = collaborative_recommendation(user_id, content_type)[:50]  # Get more CF recs
    content_recs = content_based_recommendation(
        genre_filter, year_filter, content_type, country_filter, n=50
    )  # Get more CBF recs
    
    # Create scoring dictionary
    movie_scores = {}
    
    # Process collaborative recommendations (70% weight)
    for rec in collab_recs:
        movie_scores[rec['id']] = {
            'score': (rec['rating'] / 5) * 0.7,  # Normalized CF score
            'cf_score': rec['rating'],
            'cbf_score': 0,
            'details': rec
        }
    
    # Process content recommendations (30% weight)
    for rec in content_recs:
        cbf_component = (rec['rating'] / 5) * 0.3  # Normalized CBF score
        if rec['id'] in movie_scores:
            movie_scores[rec['id']]['score'] += cbf_component
            movie_scores[rec['id']]['cbf_score'] = rec['rating']
        else:
            movie_scores[rec['id']] = {
                'score': cbf_component,
                'cf_score': 0,
                'cbf_score': rec['rating'],
                'details': rec
            }
    
    # Convert to list format with all details
    hybrid_recs = []
    for movie_id, data in movie_scores.items():
        details = data['details'].copy()
        details.update({
            'hybrid_score': data['score'] * 5,  # Scale back to 0-5 range for display
            'cf_score': data['cf_score'],
            'cbf_score': data['cbf_score']
        })
        hybrid_recs.append(details)
    
    # Sort by hybrid score
    hybrid_recs.sort(key=lambda x: x['hybrid_score'], reverse=True)
    
    # Apply some diversity to the results
    final_recs = []
    genres_seen = set()
    
    for rec in hybrid_recs:
        rec_genres = set(g.lower() for g in rec.get('genres', []))
        if not genres_seen.intersection(rec_genres) or len(final_recs) < 5:
            final_recs.append(rec)
            genres_seen.update(rec_genres)
        if len(final_recs) >= 10:
            break
    
    return final_recs[:10]


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
                recommendations = content_based_recommendation(
                    genre_filter=genre_filter,
                    year_filter=year_filter,
                    content_type=content_type,
                    country_filter=country_filter
                )
                filters_text = []
                if genre_filter:
                    filters_text.append(f"Genres: {', '.join(genre_filter)}")
                if year_filter != (1945, 2022):
                    filters_text.append(f"Years: {year_filter[0]}-{year_filter[1]}")
                if content_type != "All":
                    filters_text.append(f"Type: {content_type}")
                if country_filter:
                    filters_text.append(f"Country: {country_filter}")
                
                filter_display = f" ({', '.join(filters_text)})" if filters_text else ""
                st.subheader(f"Top Netflix TV Shows & Movie Based on Content{filter_display}")
            else:  # Hybrid
                recommendations = hybrid_recommendation(
                    user_id=user_id,
                    genre_filter=genre_filter,
                    year_filter=year_filter,
                    content_type=content_type,
                    country_filter=country_filter
                )
                filters_text = []
                if genre_filter:
                    filters_text.append(f"Genres: {', '.join(genre_filter)}")
                if year_filter != (1945, 2022):
                    filters_text.append(f"Years: {year_filter[0]}-{year_filter[1]}")
                if content_type != "All":
                    filters_text.append(f"Type: {content_type}")
                if country_filter:
                    filters_text.append(f"Country: {country_filter}")
                
                filter_display = f" ({', '.join(filters_text)})" if filters_text else ""
                st.subheader(f"Top Netflix TV Shows & Movie for User {user_id} (Hybrid Recommendations){filter_display}")
            
            if not recommendations:
                st.warning("No recommendations found matching your criteria. Try adjusting your filters.")
            else:
                # Display all recommendations
                for movie in recommendations:
                    display_movie_card(movie, recommender_type)
                
        except Exception as e:
            st.error(f"Error generating recommendations: {e}")
            st.expander("Show error details").code(str(e))

# Footer
st.markdown("---")
st.markdown("Netflix TV Shows & Movie Recommender System - LGL")
