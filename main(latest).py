import streamlit as st
import joblib
from joblib import load
import pandas as pd
import numpy as np
import os
import re
from ast import literal_eval

# --- Page Config (MUST BE FIRST STREAMLIT COMMAND) ---
st.set_page_config(layout="wide")


# Sklearn imports
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix, hstack


# --- File Loading and Checks ---
# (Rest of the file loading code remains the same as before)
required_files = ['collaborative_model.joblib', 'content_model.joblib', 'hybrid_model.joblib', 'titles.xls', 'user_interactions.xls']
missing_files = [file for file in required_files if not os.path.exists(file)]

if missing_files:
    st.error(f"Error: The following required files are missing: {', '.join(missing_files)}")
    st.stop()

try:
    titles = pd.read_csv('titles.xls', usecols=['id', 'title', 'genres', 'production_countries', 'release_year', 'type', 'description'], dtype={'id': str})
    user_interactions = pd.read_csv('user_interactions.xls', usecols=['user_id', 'id', 'rating'], dtype={'id': str})
    collaborative_model = load('collaborative_model.joblib')
    content_model_components = load('content_model.joblib')
    hybrid_model = load('hybrid_model.joblib')
    tfidf_vectorizer = content_model_components.get('tfidf_vectorizer')
    scaler = content_model_components.get('scaler')
    if not all([tfidf_vectorizer, scaler]):
         raise ValueError("Content model joblib is missing 'tfidf_vectorizer' or 'scaler'.")
except FileNotFoundError as e:
     st.error(f"File not found during loading: {e}")
     st.stop()
except Exception as e:
    st.error(f"Error loading models or data files: {e}")
    st.stop()
# --- End File Loading ---


# --- Recommendation Functions ---
def collaborative_recommendation(user_id, content_type=None):
    """Generate recommendations using collaborative filtering model (SVD)"""
    user_rated_movies = user_interactions[user_interactions['user_id'] == user_id]['id'].tolist()
    unrated_movies = titles[~titles['id'].isin(user_rated_movies)].copy()
    
    if content_type and content_type != "All":
        content_type_filter = content_type.strip().upper()  # Match case in titles df
        unrated_movies = unrated_movies[unrated_movies['type'].str.strip().str.upper() == content_type_filter]
    
    if unrated_movies.empty:
        return []
    
    predictions = []
    for _, row in unrated_movies.iterrows():
        try:
            # Ensure IDs match format (string if needed)
            pred = collaborative_model.predict(uid=user_id, iid=str(row['id']))
            predictions.append({
                "title": row['title'], 
                "rating": pred.est, 
                "id": str(row['id']),
                "description": row.get('description', 'No description available'),
                "genres": row.get('genres', 'N/A')
            })
        except Exception as e:
            continue  # Skip if prediction fails
    
    return sorted(predictions, key=lambda x: x['rating'], reverse=True)[:10]


def content_based_recommendation(genre_filter=None, year_filter=None, content_type=None, top_n=10):
    """Generate recommendations using content-based filtering aligned with the notebook."""
    filtered_content = titles.copy() # Start with the full titles data

    # 1. Initial Filtering
    if content_type and content_type != "All":
        content_type_filter = content_type.strip().upper()
        filtered_content = filtered_content[filtered_content['type'].str.strip().str.upper() == content_type_filter]

    if year_filter:
        # Allow a range as in the notebook (+- 2 years)
        year_min = year_filter - 2
        year_max = year_filter + 2
        filtered_content = filtered_content[
            (filtered_content['release_year'] >= year_min) &
            (filtered_content['release_year'] <= year_max)
        ].dropna(subset=['release_year'])


    if genre_filter:
        # Handle list-like strings in 'genres'
        def check_genre(genres_str):
            try:
                # Safely evaluate the string representation of the list
                genres_list = literal_eval(genres_str)
                # Check if the filter genre exists in the list (case-insensitive)
                return genre_filter.strip().lower() in [g.strip().lower() for g in genres_list if isinstance(g, str)]
            except (ValueError, SyntaxError, TypeError):
                # Handle cases where the format is unexpected or NaN
                 return False
        filtered_content = filtered_content[filtered_content['genres'].apply(check_genre)]


    if filtered_content.empty:
        return []

    # Ensure unique titles after filtering
    filtered_content = filtered_content.drop_duplicates(subset='id').reset_index(drop=True)

    if filtered_content.empty:
        return []


    # # 2. Preprocessing (on filtered data) - Apply only if necessary columns exist
    # text_cols_to_process = ['description']
    # for col in text_cols_to_process:
    #      if col in filtered_content.columns:
    #           # Create processed column only if it doesn't exist or needs update
    #           if f'processed_{col}' not in filtered_content.columns:
    #               filtered_content[f'processed_{col}'] = filtered_content[col].apply(preprocess_text)
    #      # else:
    #      #      st.warning(f"Column '{col}' not found for preprocessing.")


    # Combine features for TF-IDF (using original columns as notebook did)
    # Ensure NaN handling similar to notebook (fillna)
    filtered_content['text_features'] = (
        filtered_content['genres'].fillna('[]').astype(str) + " " + # Treat genres as string lists
        filtered_content['production_countries'].fillna('[]').astype(str) + " " +
        filtered_content['description'].fillna('No description available')
    )

    # Remove rows where text_features became empty after processing/fillna
    filtered_content = filtered_content[filtered_content['text_features'].str.strip() != '']

    if filtered_content.empty:
        return []

    # 3. Feature Engineering (using loaded components)
    try:
        tfidf_matrix = tfidf_vectorizer.transform(filtered_content['text_features'])
    except Exception as e:
        st.error(f"Error applying TF-IDF: {e}")
        return []

    # Scale numeric features (release_year)
    # Handle potential missing 'release_year' more robustly
    if 'release_year' not in filtered_content.columns:
        st.warning("'release_year' column missing, skipping numeric scaling.")
        numeric_sparse = csr_matrix((tfidf_matrix.shape[0], 0)) # Empty sparse matrix if no numeric features
    else:
        numeric_features = filtered_content[['release_year']].fillna(filtered_content[['release_year']].mean())
        try:
            numeric_scaled = scaler.transform(numeric_features) # Use transform, not fit_transform
            numeric_sparse = csr_matrix(numeric_scaled)
        except Exception as e:
            st.error(f"Error scaling numeric features: {e}")
            return []


    # Combine features
    try:
        combined_features = hstack([tfidf_matrix, numeric_sparse]).tocsr()
        # Handle potential NaNs introduced in sparse data (though less likely after fillna)
        combined_features.data = np.nan_to_num(combined_features.data, copy=False)
    except Exception as e:
         st.error(f"Error combining features: {e}")
         return []


    # 4. Similarity Calculation (on the combined features of filtered data)
    # Check if there's more than one item to compare
    if combined_features.shape[0] < 2:
        # If only one item matches filters, return it as the only recommendation
        if not filtered_content.empty:
            row = filtered_content.iloc[0]
            movie_id_str = str(row['id'])
            avg_rating = user_interactions[user_interactions['id'] == movie_id_str]['rating'].mean()
            display_rating = avg_rating if pd.notna(avg_rating) else 3.0 # Fallback rating if no user data
            return [{
                 "title": row['title'],
                 "rating": float(display_rating),
                 "id": movie_id_str,
                 "description": row.get('description', 'N/A'),
                 "genres": row.get('genres', 'N/A')
            }]
        else:
            return [] # No items to recommend

    cosine_sim = cosine_similarity(combined_features, combined_features)

    # Create a similarity DataFrame with the filtered content's index
    similarity_df = pd.DataFrame(cosine_sim, index=filtered_content.index, columns=filtered_content.index)

    # 5. Recommendation Generation
    # Get average similarity scores across all filtered items (as a proxy for user preference)
    # We take the mean similarity of each item to all *other* items within the filtered set
    np.fill_diagonal(similarity_df.values, 0) # Set diagonal to 0 to exclude self-similarity
    avg_similarity_scores = similarity_df.mean(axis=1)

    # Sort items by their average similarity score within the filtered group
    recommended_indices = avg_similarity_scores.sort_values(ascending=False).index

    # Get the top N recommended items' details from the filtered data
    recommendations_df = filtered_content.loc[recommended_indices].head(top_n)

    # Format output
    recommendations = []
    processed_ids = set() # Keep track of added movie IDs to avoid duplicates if index issue occurs
    for _, row in recommendations_df.iterrows():
         movie_id_str = str(row['id']) # Ensure ID is string for matching
         if movie_id_str in processed_ids:
              continue # Skip if already added

         avg_rating = user_interactions[user_interactions['id'] == movie_id_str]['rating'].mean()
         # Use the calculated average similarity score if no user ratings exist (more relevant than fixed fallback)
         # Ensure the index `row.name` exists in `avg_similarity_scores`
         similarity_score = avg_similarity_scores.get(row.name, 0.0)
         display_rating = avg_rating if pd.notna(avg_rating) else similarity_score

         recommendations.append({
             "title": row['title'],
             "rating": float(display_rating), # Use average rating or similarity score
             "id": movie_id_str,
             "description": row.get('description', 'N/A'), # Add description
             "genres": row.get('genres', 'N/A') # Add genres
         })
         processed_ids.add(movie_id_str)

    # Sort final list by display_rating
    return sorted(recommendations, key=lambda x: x['rating'], reverse=True)


def hybrid_recommendation(user_id, genre_filter=None, year_filter=None, content_type=None, top_n=10):
    """Generate recommendations using hybrid approach (collaborative + content-based)"""
    # Extract weights (assuming they are stored correctly)
    # Default weights if not found in the model file
    default_weights = (0.5, 0.5) # Example: equal weighting
    weights = hybrid_model.get('weights', default_weights)
    collab_weight, content_weight = weights

    # Get collaborative filtering recommendations
    collab_recs = collaborative_recommendation(user_id, content_type) # Gets top 10

    # Get content-based recommendations
    # Request more content recs initially to allow for better merging
    content_recs = content_based_recommendation(genre_filter, year_filter, content_type, top_n=top_n * 2)

    # Combine recommendations
    combined_scores = {}
    item_details = {} # Store details like description, genres, ID

    # Process collaborative recommendations (predicted ratings)
    for rec in collab_recs:
         title = rec['title']
         # Ensure ID is string
         item_id = str(rec.get('id', 'N/A'))
         if item_id == 'N/A': continue # Skip if ID missing

         combined_scores[title] = rec['rating'] * collab_weight
         if title not in item_details:
             details = titles[titles['id'] == item_id].iloc[0] if not titles[titles['id'] == item_id].empty else None
             if details is not None:
                 item_details[title] = {
                     'id': item_id,
                     'description': details.get('description', 'N/A'),
                     'genres': details.get('genres', 'N/A')}


    # Process content recommendations (average/similarity ratings)
    for rec in content_recs:
        title = rec['title']
        item_id = str(rec.get('id', 'N/A'))
        if item_id == 'N/A': continue # Skip if ID missing

        score_contribution = rec['rating'] * content_weight
        if title in combined_scores:
            combined_scores[title] += score_contribution
        else:
            combined_scores[title] = score_contribution

        if title not in item_details:
             item_details[title] = {
                  'id': item_id,
                  'description': rec.get('description', 'N/A'),
                  'genres': rec.get('genres', 'N/A')}


    # Convert back to list of dictionaries including details
    hybrid_recs = []
    for title, score in combined_scores.items():
         details = item_details.get(title, {'id': 'N/A', 'description': 'N/A', 'genres': 'N/A'})
         hybrid_recs.append({
              "title": title,
              "rating": score, # This is the combined score
              "id": details['id'],
              "description": details['description'],
              "genres": details['genres']
         })


    # Sort by combined score and return top N
    return sorted(hybrid_recs, key=lambda x: x['rating'], reverse=True)[:top_n]
# --- End Recommendation Functions ---


# --- Streamlit UI ---
# (Streamlit UI code remains the same as the previous correct version)
# Title should come after page config
st.title("🎬 Movie Recommendation System")

# Sidebar for controls
st.sidebar.header("Configure Recommendations")
recommender_type = st.sidebar.selectbox("Select Recommendation Method", ["Hybrid", "Collaborative Filtering", "Content-Based"])

# Initialize variables
user_id = None
genre_filter = "" # Initialize as empty string for selectbox default
year_filter = None
content_type = "All" # Default to All


# Conditional inputs based on recommender type
if recommender_type in ["Collaborative Filtering", "Hybrid"]:
    # Ensure user_ids from the interactions file are used for the range
    try:
        min_user_id = int(user_interactions['user_id'].min())
        max_user_id = int(user_interactions['user_id'].max())
        user_id = st.sidebar.number_input(f"Enter your User ID ({min_user_id}-{max_user_id})", min_value=min_user_id, max_value=max_user_id, value=min_user_id)
    except Exception as e:
        st.sidebar.error(f"Could not determine User ID range: {e}")
        user_id = st.sidebar.number_input("Enter your User ID", value=1) # Fallback


if recommender_type in ["Content-Based", "Hybrid"]:
    content_type = st.sidebar.radio("Filter by Type", ["All", "Movie", "Show"], index=0) # Default 'All'

    # Dynamic Genre List from data
    try:
         all_genres = set()
         # Handle potential string representations of lists more robustly
         def extract_genres(genres_str):
             try:
                 evaluated_list = literal_eval(genres_str)
                 if isinstance(evaluated_list, list):
                     return {g.strip().lower() for g in evaluated_list if isinstance(g, str)}
                 return set()
             except:
                 return set() # Return empty set on error

         # Apply the robust extraction function
         genre_sets = titles['genres'].dropna().apply(extract_genres)
         for genre_set in genre_sets:
             all_genres.update(genre_set)

         genre_options = [""] + sorted(list(all_genres)) # Add empty option for no filter
         genre_filter = st.sidebar.selectbox("Choose a genre (optional):", genre_options, index=0)
    except Exception as e:
         st.sidebar.warning(f"Could not parse genres dynamically: {e}. Using default list.")
         genre_options = ["", "action", "animation", "comedy", "crime", "documentation", "drama", "european", "family", "fantasy", "history", "horror", "music", "reality", "romance", "scifi", "sport", "thriller", "war", "western"]
         genre_filter = st.sidebar.selectbox("Choose a genre (optional):", genre_options, index=0)


    filter_by_year = st.sidebar.checkbox("Filter by release year")
    if filter_by_year:
         try:
             min_year = int(titles['release_year'].dropna().min())
             max_year = int(titles['release_year'].dropna().max())
             # Set default value within the range, e.g., midpoint or recent year
             default_year = max(min_year, min(max_year, 2010))
             year_slider_value = st.sidebar.slider("Select approximate release year (+- 2 years)", min_year, max_year, default_year)
             year_filter = year_slider_value
         except Exception as e:
             st.sidebar.error(f"Could not determine year range: {e}")
             year_filter = st.sidebar.number_input("Enter release year", value=2010) # Fallback
    else:
        year_filter = None # Set to None if checkbox is unchecked

# Button to trigger recommendations
if st.sidebar.button("Get Recommendations"):
    recommendations = []
    error_message = None
    try:
        # Input validation
        if recommender_type in ["Collaborative Filtering", "Hybrid"] and user_id is None:
            error_message = "Please enter a User ID for Collaborative or Hybrid recommendations."
        
        # Handle empty string genre filter explicitly
        current_genre_filter = genre_filter if genre_filter else None

        # Check if any filters are active for Content-Based/Hybrid
        no_content_filters = not current_genre_filter and not year_filter and content_type == "All"

        if recommender_type == "Content-Based" and no_content_filters:
            st.info("Showing general Content-Based recommendations (top rated overall). Add filters for more specific results.")
            # Fallback: Show top-rated overall from the titles dataset
            top_rated_ids = user_interactions.groupby('id')['rating'].mean().nlargest(10).index.tolist()
            recommendations_df = titles[titles['id'].isin(top_rated_ids)].copy()
            
            # Add the average rating to the DataFrame for sorting/display
            avg_ratings_map = user_interactions[user_interactions['id'].isin(top_rated_ids)].groupby('id')['rating'].mean()
            recommendations_df['avg_rating'] = recommendations_df['id'].map(avg_ratings_map)

            recommendations = [{
                "title": row['title'],
                "rating": row['avg_rating'] if pd.notna(row.get('avg_rating')) else 0.0,
                "id": str(row['id']),
                "description": row.get('description', 'N/A'),
                "genres": row.get('genres', 'N/A')
            } for _, row in recommendations_df.iterrows()]
            
            # Handle potential NaN ratings before sorting
            recommendations = sorted([rec for rec in recommendations if pd.notna(rec['rating'])], 
                                    key=lambda x: x['rating'], reverse=True)

        elif recommender_type == "Hybrid" and no_content_filters:
            st.info("Showing Hybrid recommendations based primarily on User ID collaborative score. Add content filters for refinement.")
            recommendations = hybrid_recommendation(user_id, None, None, "All")

        elif recommender_type == "Collaborative Filtering":
            recommendations = collaborative_recommendation(user_id, content_type)
        elif recommender_type == "Content-Based":
            recommendations = content_based_recommendation(current_genre_filter, year_filter, content_type)
        elif recommender_type == "Hybrid":
            recommendations = hybrid_recommendation(user_id, current_genre_filter, year_filter, content_type)

    except Exception as e:
        error_message = f"An error occurred during recommendation generation: {e}"
        import traceback
        st.error(error_message)
        st.code(traceback.format_exc())  # Show detailed error in app

    # Display Results or Errors
    st.header("Recommendations")
    if error_message and not recommendations:  # Show error only if no recommendations were generated
        st.error(error_message)
    elif not recommendations:
        st.warning("No recommendations found matching your criteria. Please try adjusting the filters.")
    else:
        st.success(f"Top {len(recommendations)} recommendations using **{recommender_type}** method:")

        # Determine rating label based on method
        if recommender_type == "Collaborative Filtering":
            rating_label = "Predicted Rating"
        elif recommender_type == "Content-Based":
            rating_label = "Avg. Rating / Similarity"
        else:  # Hybrid
            rating_label = "Combined Score"

        # Display in columns for better layout
        cols = st.columns(2)  # Create 2 columns
        for i, movie in enumerate(recommendations):
            # Safely format rating
            rating_val = movie.get('rating')
            rating_text = f"{rating_val:.2f}" if isinstance(rating_val, (float, np.float64, int)) else "N/A"

            with cols[i % 2]:  # Alternate columns
                st.subheader(f"{i+1}. {movie.get('title', 'Unknown Title')}")
                st.markdown(f"**{rating_label}:** {rating_text}")

                # Display Genres if available
                genres_data = movie.get('genres')
                if genres_data and genres_data != 'N/A':
                    try:
                        # Attempt to display genres nicely, whether list or string list
                        if isinstance(genres_data, str):
                            genres_list = literal_eval(genres_data)
                        else:
                            genres_list = genres_data  # Assume it's already a list
                        if isinstance(genres_list, list):
                            st.markdown(f"**Genres:** {', '.join(g for g in genres_list if isinstance(g, str))}")
                        else:
                            st.markdown(f"**Genres:** {genres_data}")  # Fallback to raw string
                    except:
                        st.markdown(f"**Genres:** {genres_data}")  # Fallback if eval fails

                # Display Description in expander if available
                description_data = movie.get('description')
                if description_data and description_data != 'N/A' and description_data != 'No description available':
                    with st.expander("Description"):
                        st.write(description_data)

                st.write("---")  # Separator


# Optional: Add instructions or info
st.sidebar.markdown("---")
st.sidebar.info("Select a recommendation method and adjust filters as needed. Click 'Get Recommendations' to see results.")
