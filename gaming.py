import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.decomposition import TruncatedSVD
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="GameFinder AI",
    page_icon="🎮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        padding: 1rem 0;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #667eea;
    }
    .recommendation-card {
        background-color: #ffffff;
        padding: 1rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 0.5rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'user_preferences' not in st.session_state:
    st.session_state.user_preferences = {
        'liked_games': [],
        'disliked_games': [],
        'preferred_genres': [],
        'preferred_platforms': []
    }

if 'recommendation_history' not in st.session_state:
    st.session_state.recommendation_history = []

# Load and preprocess data
@st.cache_data
def load_data(file):
    """Load and preprocess the dataset with robust error handling"""
    try:
        if file is not None:
            df = pd.read_csv(file)
        else:
            return None
        
        # Validate required columns
        required_columns = ['Name', 'Platform', 'Genre', 'Publisher', 'Global_Sales']
        missing_columns = set(required_columns) - set(df.columns)
        
        if missing_columns:
            st.error(f"❌ Missing required columns: {missing_columns}")
            return None
        
        # Data cleaning
        df = df.dropna(subset=['Name', 'Platform', 'Genre', 'Publisher'])
        
        # Handle Year column
        if 'Year' in df.columns:
            df['Year'] = pd.to_numeric(df['Year'], errors='coerce')
            df['Year'] = df['Year'].fillna(df['Year'].median())
        else:
            df['Year'] = 2010  # Default year
        
        # Handle sales columns
        sales_columns = ['NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales', 'Global_Sales']
        for col in sales_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        
        # Create combined features for better matching
        df['Game_ID'] = df.groupby('Name').cumcount()
        df['Full_Name'] = df['Name'] + ' (' + df['Platform'] + ')'
        
        # Remove duplicates
        df = df.drop_duplicates(subset=['Name', 'Platform'])
        
        # Create popularity score
        df['Popularity_Score'] = (
            df['Global_Sales'] * 0.6 +
            (2025 - df['Year']) * 0.01 +  # Recency bonus
            df.groupby('Name')['Platform'].transform('count') * 0.1  # Multi-platform bonus
        )
        
        return df
        
    except Exception as e:
        st.error(f"❌ Error loading data: {str(e)}")
        return None

@st.cache_data
def build_advanced_features(df):
    """Build comprehensive feature matrix for recommendations"""
    try:
        # Create feature DataFrame
        feature_df = df.copy()
        
        # Numerical features
        numerical_features = ['Global_Sales', 'Year']
        if 'NA_Sales' in df.columns:
            numerical_features.extend(['NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales'])
        
        # Categorical features
        categorical_features = ['Genre', 'Platform', 'Publisher']
        
        # Scale numerical features
        scaler = StandardScaler()
        scaled_numerical = scaler.fit_transform(feature_df[numerical_features])
        
        # One-hot encode categorical features
        encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
        encoded_categorical = encoder.fit_transform(feature_df[categorical_features])
        
        # Combine features
        feature_matrix = np.hstack([encoded_categorical, scaled_numerical])
        
        return feature_matrix, encoder, scaler, categorical_features, numerical_features
        
    except Exception as e:
        st.error(f"❌ Error building features: {str(e)}")
        return None, None, None, None, None

@st.cache_data
def compute_similarity_matrices(_feature_matrix):
    """Compute multiple similarity matrices for hybrid recommendations"""
    cosine_sim = cosine_similarity(_feature_matrix)
    
    # Also compute euclidean distance-based similarity
    euclidean_dist = euclidean_distances(_feature_matrix)
    euclidean_sim = 1 / (1 + euclidean_dist)  # Convert distance to similarity
    
    # Hybrid similarity (weighted average)
    hybrid_sim = 0.7 * cosine_sim + 0.3 * euclidean_sim
    
    return cosine_sim, euclidean_sim, hybrid_sim

def get_game_details(df, game_name):
    """Get detailed information about a game"""
    game_data = df[df['Full_Name'] == game_name]
    if game_data.empty:
        game_data = df[df['Name'].str.contains(game_name, case=False, na=False)]
    
    if not game_data.empty:
        return game_data.iloc[0]
    return None

def content_based_recommendations(game_name, df, similarity_matrix, top_n=10, exclude_same_name=True):
    """Advanced content-based recommendations"""
    # Find the game
    matches = df[df['Full_Name'] == game_name]
    
    if matches.empty:
        # Try fuzzy matching
        matches = df[df['Name'].str.contains(game_name.split('(')[0].strip(), case=False, na=False)]
        if matches.empty:
            return pd.DataFrame(), "Game not found"
    
    idx = matches.index[0]
    game_info = df.iloc[idx]
    
    # Get similarity scores
    sim_scores = list(enumerate(similarity_matrix[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    
    # Filter recommendations
    recommendations = []
    for i, score in sim_scores[1:]:  # Skip the game itself
        candidate = df.iloc[i]
        
        # Exclude same game on different platforms if requested
        if exclude_same_name and candidate['Name'] == game_info['Name']:
            continue
        
        recommendations.append((i, score))
        
        if len(recommendations) >= top_n:
            break
    
    # Get recommended games
    game_indices = [i[0] for i in recommendations]
    similarity_scores = [i[1] for i in recommendations]
    
    result_df = df.iloc[game_indices][['Full_Name', 'Name', 'Platform', 'Genre', 'Publisher', 'Year', 'Global_Sales']].copy()
    result_df['Similarity_Score'] = similarity_scores
    result_df['Match_Percentage'] = (result_df['Similarity_Score'] * 100).round(1)
    
    return result_df, None

def popularity_based_recommendations(df, genre=None, platform=None, publisher=None, 
                                    year_range=None, top_n=10):
    """Advanced popularity-based recommendations with multiple filters"""
    filtered_df = df.copy()
    
    # Apply filters
    if genre and genre != "All":
        filtered_df = filtered_df[filtered_df['Genre'] == genre]
    
    if platform and platform != "All":
        filtered_df = filtered_df[filtered_df['Platform'] == platform]
    
    if publisher and publisher != "All":
        filtered_df = filtered_df[filtered_df['Publisher'] == publisher]
    
    if year_range:
        filtered_df = filtered_df[
            (filtered_df['Year'] >= year_range[0]) & 
            (filtered_df['Year'] <= year_range[1])
        ]
    
    if filtered_df.empty:
        return pd.DataFrame()
    
    # Sort by popularity score
    popular = filtered_df.sort_values('Popularity_Score', ascending=False).head(top_n)
    
    return popular[['Full_Name', 'Name', 'Platform', 'Genre', 'Publisher', 'Year', 'Global_Sales', 'Popularity_Score']]

def hybrid_recommendations(df, similarity_matrix, user_prefs, top_n=10):
    """Hybrid recommendations based on user preferences"""
    if not user_prefs['liked_games']:
        return pd.DataFrame(), "Please like some games first to get personalized recommendations"
    
    # Get indices of liked games
    liked_indices = []
    for game in user_prefs['liked_games']:
        match = df[df['Full_Name'] == game]
        if not match.empty:
            liked_indices.append(match.index[0])
    
    if not liked_indices:
        return pd.DataFrame(), "Could not find liked games in dataset"
    
    # Aggregate similarity scores from liked games
    aggregated_scores = np.mean(similarity_matrix[liked_indices], axis=0)
    
    # Penalize disliked games
    if user_prefs['disliked_games']:
        for game in user_prefs['disliked_games']:
            match = df[df['Full_Name'] == game]
            if not match.empty:
                idx = match.index[0]
                aggregated_scores[idx] = -1
    
    # Boost preferred genres and platforms
    genre_boost = 1.2 if user_prefs['preferred_genres'] else 1.0
    platform_boost = 1.15 if user_prefs['preferred_platforms'] else 1.0
    
    for i, row in df.iterrows():
        if row['Genre'] in user_prefs['preferred_genres']:
            aggregated_scores[i] *= genre_boost
        if row['Platform'] in user_prefs['preferred_platforms']:
            aggregated_scores[i] *= platform_boost
    
    # Get top recommendations
    sim_scores = list(enumerate(aggregated_scores))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    
    # Exclude already liked games
    recommendations = []
    for i, score in sim_scores:
        if df.iloc[i]['Full_Name'] not in user_prefs['liked_games']:
            recommendations.append((i, score))
            if len(recommendations) >= top_n:
                break
    
    game_indices = [i[0] for i in recommendations]
    scores = [i[1] for i in recommendations]
    
    result_df = df.iloc[game_indices][['Full_Name', 'Name', 'Platform', 'Genre', 'Publisher', 'Year', 'Global_Sales']].copy()
    result_df['Recommendation_Score'] = scores
    result_df['Match_Percentage'] = (MinMaxScaler().fit_transform(result_df[['Recommendation_Score']]) * 100).round(1)
    
    return result_df, None

def discover_recommendations(df, similarity_matrix, top_n=10):
    """Discover hidden gems and diverse recommendations"""
    # Find games with good quality but lower popularity
    df_copy = df.copy()
    
    # Create quality score (good reviews but not necessarily popular)
    df_copy['Quality_Score'] = (
        df_copy['Global_Sales'].apply(lambda x: min(x, 10)) +  # Cap sales impact
        (2025 - df_copy['Year']) * -0.05 +  # Prefer older hidden gems
        10  # Base score
    )
    
    # Filter for "hidden gems" (decent sales but not top sellers)
    hidden_gems = df_copy[
        (df_copy['Global_Sales'] > df_copy['Global_Sales'].quantile(0.3)) &
        (df_copy['Global_Sales'] < df_copy['Global_Sales'].quantile(0.85))
    ]
    
    if hidden_gems.empty:
        hidden_gems = df_copy
    
    # Sample diverse genres
    diverse_recommendations = hidden_gems.groupby('Genre').apply(
        lambda x: x.nlargest(2, 'Quality_Score')
    ).reset_index(drop=True)
    
    result = diverse_recommendations.sample(min(top_n, len(diverse_recommendations)))
    
    return result[['Full_Name', 'Name', 'Platform', 'Genre', 'Publisher', 'Year', 'Global_Sales']]

def search_games(df, query):
    """Search for games by name"""
    if not query:
        return df.head(20)
    
    # Search in name
    mask = df['Name'].str.contains(query, case=False, na=False)
    results = df[mask]
    
    return results[['Full_Name', 'Name', 'Platform', 'Genre', 'Publisher', 'Year', 'Global_Sales']].head(20)

def create_genre_distribution_chart(df):
    """Create interactive genre distribution chart"""
    genre_counts = df['Genre'].value_counts().head(10)
    
    fig = px.bar(
        x=genre_counts.values,
        y=genre_counts.index,
        orientation='h',
        title='Top 10 Genres in Dataset',
        labels={'x': 'Number of Games', 'y': 'Genre'},
        color=genre_counts.values,
        color_continuous_scale='Viridis'
    )
    fig.update_layout(showlegend=False, height=400)
    
    return fig

def create_platform_chart(df):
    """Create platform distribution chart"""
    platform_counts = df['Platform'].value_counts().head(10)
    
    fig = px.pie(
        values=platform_counts.values,
        names=platform_counts.index,
        title='Top 10 Platforms in Dataset',
        hole=0.4
    )
    fig.update_layout(height=400)
    
    return fig

def create_sales_timeline(df):
    """Create sales over time chart"""
    yearly_sales = df.groupby('Year')['Global_Sales'].sum().reset_index()
    
    fig = px.line(
        yearly_sales,
        x='Year',
        y='Global_Sales',
        title='Global Sales Over Time',
        labels={'Global_Sales': 'Total Sales (millions)', 'Year': 'Year'}
    )
    fig.update_layout(height=400)
    
    return fig

# Main application
def main():
    # Header
    st.markdown('<h1 class="main-header">🎮 GameFinder AI</h1>', unsafe_allow_html=True)
    st.markdown("### Your Ultimate Video Game Recommendation Engine")
    st.markdown("---")
    
    # Sidebar - File Upload and Configuration
    with st.sidebar:
        st.header("📁 Data Upload")
        uploaded_file = st.file_uploader(
            "Upload Video Game Dataset (CSV)",
            type=['csv'],
            help="Upload a CSV file with columns: Name, Platform, Genre, Publisher, Global_Sales"
        )
        
        st.markdown("---")
        
        # Sample data option
        if st.checkbox("Use Sample Data", value=True if uploaded_file is None else False):
            st.info("📝 Please upload your vgsales.csv file to get started!")
            uploaded_file = None
        
        st.markdown("---")
        st.markdown("### 🎯 About")
        st.markdown("""
        **GameFinder AI** uses advanced machine learning algorithms to recommend games based on:
        - Content similarity
        - User preferences
        - Popularity metrics
        - Hybrid approaches
        """)
    
    # Load data
    if uploaded_file is None:
        st.warning("⚠️ Please upload a video game dataset CSV file to begin!")
        st.info("""
        **Required CSV columns:**
        - Name
        - Platform
        - Genre
        - Publisher
        - Global_Sales
        - Year (optional)
        - Regional sales columns (optional)
        """)
        st.stop()
    
    df = load_data(uploaded_file)
    
    if df is None or df.empty:
        st.error("❌ Could not load data. Please check your file format.")
        st.stop()
    
    # Build features
    with st.spinner("🔧 Building recommendation engine..."):
        feature_matrix, encoder, scaler, cat_features, num_features = build_advanced_features(df)
        
        if feature_matrix is None:
            st.error("❌ Error building features")
            st.stop()
        
        cosine_sim, euclidean_sim, hybrid_sim = compute_similarity_matrices(feature_matrix)
    
    # Dataset Overview
    st.success(f"✅ Loaded {len(df)} games successfully!")
    
    # Metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Games", f"{len(df):,}")
    with col2:
        st.metric("Genres", f"{df['Genre'].nunique()}")
    with col3:
        st.metric("Platforms", f"{df['Platform'].nunique()}")
    with col4:
        st.metric("Publishers", f"{df['Publisher'].nunique()}")
    
    st.markdown("---")
    
    # Main tabs
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "🎯 Content-Based",
        "🔥 Popular Games",
        "💝 Personalized",
        "🔍 Discover",
        "📊 Analytics",
        "⚙️ Preferences"
    ])
    
    # Tab 1: Content-Based Recommendations
    with tab1:
        st.header("🎯 Content-Based Recommendations")
        st.write("Find games similar to ones you already love!")
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            search_query = st.text_input("🔍 Search for a game", placeholder="Type game name...")
            
            if search_query:
                search_results = search_games(df, search_query)
                if not search_results.empty:
                    game_options = search_results['Full_Name'].tolist()
                else:
                    game_options = sorted(df['Full_Name'].unique())
            else:
                game_options = sorted(df['Full_Name'].unique())
            
            selected_game = st.selectbox(
                "Select a game",
                game_options,
                help="Choose a game to get similar recommendations"
            )
        
        with col2:
            top_n = st.slider("Number of recommendations", 5, 20, 10)
            similarity_type = st.selectbox(
                "Similarity method",
                ["Hybrid", "Cosine", "Euclidean"],
                help="Choose the algorithm for similarity calculation"
            )
        
        exclude_same = st.checkbox("Exclude same game on different platforms", value=True)
        
        if st.button("🎮 Get Recommendations", key="content_based", type="primary"):
            sim_matrix = hybrid_sim if similarity_type == "Hybrid" else (
                cosine_sim if similarity_type == "Cosine" else euclidean_sim
            )
            
            with st.spinner("Finding similar games..."):
                recommendations, error = content_based_recommendations(
                    selected_game, df, sim_matrix, top_n, exclude_same
                )
                
                if error:
                    st.error(f"❌ {error}")
                elif not recommendations.empty:
                    # Show selected game info
                    game_info = get_game_details(df, selected_game)
                    if game_info is not None:
                        st.markdown("### 📌 Selected Game")
                        info_col1, info_col2, info_col3, info_col4 = st.columns(4)
                        info_col1.metric("Genre", game_info['Genre'])
                        info_col2.metric("Platform", game_info['Platform'])
                        info_col3.metric("Year", int(game_info['Year']))
                        info_col4.metric("Sales (M)", f"{game_info['Global_Sales']:.2f}")
                    
                    st.markdown("### 🎯 Recommended Games")
                    
                    # Display recommendations
                    for idx, row in recommendations.iterrows():
                        with st.container():
                            col1, col2, col3 = st.columns([3, 1, 1])
                            
                            with col1:
                                st.markdown(f"**{row['Name']}**")
                                st.caption(f"{row['Genre']} | {row['Platform']} | {row['Publisher']}")
                            
                            with col2:
                                st.metric("Match", f"{row['Match_Percentage']}%")
                            
                            with col3:
                                st.metric("Sales (M)", f"{row['Global_Sales']:.2f}")
                            
                            st.markdown("---")
                    
                    # Export option
                    csv = recommendations.to_csv(index=False)
                    st.download_button(
                        "📥 Download Recommendations",
                        csv,
                        "recommendations.csv",
                        "text/csv",
                        key='download-csv'
                    )
                else:
                    st.warning("No recommendations found")
    
    # Tab 2: Popularity-Based Recommendations
    with tab2:
        st.header("🔥 Popular Games")
        st.write("Discover the most popular games with advanced filters!")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            genres = ["All"] + sorted(df['Genre'].unique().tolist())
            selected_genre = st.selectbox("Genre", genres)
        
        with col2:
            platforms = ["All"] + sorted(df['Platform'].unique().tolist())
            selected_platform = st.selectbox("Platform", platforms)
        
        with col3:
            publishers = ["All"] + sorted(df['Publisher'].unique().tolist())[:50]  # Limit for performance
            selected_publisher = st.selectbox("Publisher", publishers)
        
        # Year range filter
        min_year = int(df['Year'].min())
        max_year = int(df['Year'].max())
        year_range = st.slider(
            "Year Range",
            min_year,
            max_year,
            (min_year, max_year)
        )
        
        top_n_pop = st.slider("Number of results", 5, 30, 10, key="popular_slider")
        
        if st.button("🔥 Show Popular Games", key="popularity", type="primary"):
            with st.spinner("Finding popular games..."):
                recommendations = popularity_based_recommendations(
                    df,
                    selected_genre if selected_genre != "All" else None,
                    selected_platform if selected_platform != "All" else None,
                    selected_publisher if selected_publisher != "All" else None,
                    year_range,
                    top_n_pop
                )
                
                if not recommendations.empty:
                    st.markdown("### 🏆 Top Games")
                    
                    for idx, row in recommendations.iterrows():
                        with st.container():
                            col1, col2, col3, col4 = st.columns([3, 1, 1, 1])
                            
                            with col1:
                                st.markdown(f"**{row['Name']}**")
                                st.caption(f"{row['Genre']} | {row['Platform']} | {row['Publisher']}")
                            
                            with col2:
                                st.metric("Year", int(row['Year']))
                            
                            with col3:
                                st.metric("Sales (M)", f"{row['Global_Sales']:.2f}")
                            
                            with col4:
                                st.metric("Score", f"{row['Popularity_Score']:.1f}")
                            
                            st.markdown("---")
                    
                    # Export
                    csv = recommendations.to_csv(index=False)
                    st.download_button(
                        "📥 Download Results",
                        csv,
                        "popular_games.csv",
                        "text/csv",
                        key='download-popular'
                    )
                else:
                    st.warning("No games found with selected filters")
    
    # Tab 3: Personalized Recommendations
    with tab3:
        st.header("💝 Personalized Recommendations")
        st.write("Get recommendations based on your preferences!")
        
        if not st.session_state.user_preferences['liked_games']:
            st.info("👉 Go to the **Preferences** tab to add games you like first!")
        else:
            st.success(f"✅ You have liked {len(st.session_state.user_preferences['liked_games'])} games")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Your Liked Games:**")
                for game in st.session_state.user_preferences['liked_games'][:5]:
                    st.write(f"• {game}")
                if len(st.session_state.user_preferences['liked_games']) > 5:
                    st.write(f"... and {len(st.session_state.user_preferences['liked_games']) - 5} more")
            
            with col2:
                if st.session_state.user_preferences['preferred_genres']:
                    st.markdown("**Preferred Genres:**")
                    st.write(", ".join(st.session_state.user_preferences['preferred_genres']))
        
        top_n_hybrid = st.slider("Number of recommendations", 5, 20, 10, key="hybrid_slider")
        
        if st.button("💝 Get Personalized Recommendations", key="hybrid", type="primary"):
            with st.spinner("Creating personalized recommendations..."):
                recommendations, error = hybrid_recommendations(
                    df,
                    hybrid_sim,
                    st.session_state.user_preferences,
                    top_n_hybrid
                )
                
                if error:
                    st.error(f"❌ {error}")
                elif not recommendations.empty:
                    st.markdown("### 🎯 Your Personalized Picks")
                    
                    for idx, row in recommendations.iterrows():
                        with st.container():
                            col1, col2, col3 = st.columns([3, 1, 1])
                            
                            with col1:
                                st.markdown(f"**{row['Name']}**")
                                st.caption(f"{row['Genre']} | {row['Platform']} | {row['Publisher']}")
                            
                            with col2:
                                st.metric("Match", f"{row['Match_Percentage']}%")
                            
                            with col3:
                                st.metric("Sales (M)", f"{row['Global_Sales']:.2f}")
                            
                            st.markdown("---")
                    
                    csv = recommendations.to_csv(index=False)
                    st.download_button(
                        "📥 Download Recommendations",
                        csv,
                        "personalized_recommendations.csv",
                        "text/csv",
                        key='download-personalized'
                    )
    
    # Tab 4: Discover
    with tab4:
        st.header("🔍 Discover Hidden Gems")
        st.write("Find lesser-known games that match your taste!")
        
        st.info("This algorithm finds quality games that aren't necessarily the most popular.")
        
        discover_n = st.slider("Number of discoveries", 5, 20, 10, key="discover_slider")
        
        if st.button("🔍 Discover Games", key="discover", type="primary"):
            with st.spinner("Searching for hidden gems..."):
                recommendations = discover_recommendations(df, hybrid_sim, discover_n)
                
                if not recommendations.empty:
                    st.markdown("### 💎 Hidden Gems")
                    
                    for idx, row in recommendations.iterrows():
                        with st.container():
                            col1, col2, col3 = st.columns([3, 1, 1])
                            
                            with col1:
                                st.markdown(f"**{row['Name']}**")
                                st.caption(f"{row['Genre']} | {row['Platform']} | {row['Publisher']}")
                            
                            with col2:
                                st.metric("Year", int(row['Year']))
                            
                            with col3:
                                st.metric("Sales (M)", f"{row['Global_Sales']:.2f}")
                            
                            st.markdown("---")
                    
                    csv = recommendations.to_csv(index=False)
                    st.download_button(
                        "📥 Download Discoveries",
                        csv,
                        "hidden_gems.csv",
                        "text/csv",
                        key='download-discover'
                    )
    
    # Tab 5: Analytics
    with tab5:
        st.header("📊 Dataset Analytics")
        
        tab5_1, tab5_2, tab5_3 = st.tabs(["Genre Analysis", "Platform Analysis", "Trends"])
        
        with tab5_1:
            fig = create_genre_distribution_chart(df)
            st.plotly_chart(fig, use_container_width=True)
            
            # Top games by genre
            st.markdown("### Top Game by Each Genre")
            top_by_genre = df.loc[df.groupby('Genre')['Global_Sales'].idxmax()]
            st.dataframe(
                top_by_genre[['Genre', 'Name', 'Platform', 'Global_Sales']].sort_values('Global_Sales', ascending=False),
                use_container_width=True
            )
        
        with tab5_2:
            fig = create_platform_chart(df)
            st.plotly_chart(fig, use_container_width=True)
            
            # Platform sales comparison
            platform_sales = df.groupby('Platform')['Global_Sales'].sum().sort_values(ascending=False).head(10)
            st.markdown("### Top 10 Platforms by Total Sales")
            st.bar_chart(platform_sales)
        
        with tab5_3:
            fig = create_sales_timeline(df)
            st.plotly_chart(fig, use_container_width=True)
            
            # Year statistics
            st.markdown("### Games by Year")
            games_by_year = df['Year'].value_counts().sort_index()
            st.line_chart(games_by_year)
    
    # Tab 6: Preferences
    with tab6:
        st.header("⚙️ User Preferences")
        st.write("Customize your preferences for better recommendations!")
        
        st.markdown("### 👍 Add Liked Games")
        
        search_like = st.text_input("Search for games to like", key="search_like")
        
        if search_like:
            search_results = search_games(df, search_like)
            if not search_results.empty:
                game_to_like = st.selectbox(
                    "Select a game to add to your likes",
                    search_results['Full_Name'].tolist(),
                    key="game_to_like"
                )
                
                if st.button("👍 Add to Liked Games"):
                    if game_to_like not in st.session_state.user_preferences['liked_games']:
                        st.session_state.user_preferences['liked_games'].append(game_to_like)
                        
                        # Auto-add genre and platform preferences
                        game_data = df[df['Full_Name'] == game_to_like].iloc[0]
                        if game_data['Genre'] not in st.session_state.user_preferences['preferred_genres']:
                            st.session_state.user_preferences['preferred_genres'].append(game_data['Genre'])
                        if game_data['Platform'] not in st.session_state.user_preferences['preferred_platforms']:
                            st.session_state.user_preferences['preferred_platforms'].append(game_data['Platform'])
                        
                        st.success(f"✅ Added {game_to_like} to your liked games!")
                        st.rerun()
                    else:
                        st.warning("You already liked this game!")
        
        st.markdown("---")
        
        # Display current preferences
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 💚 Your Liked Games")
            if st.session_state.user_preferences['liked_games']:
                for i, game in enumerate(st.session_state.user_preferences['liked_games']):
                    cols = st.columns([4, 1])
                    cols[0].write(f"{i+1}. {game}")
                    if cols[1].button("❌", key=f"remove_like_{i}"):
                        st.session_state.user_preferences['liked_games'].remove(game)
                        st.rerun()
            else:
                st.info("No liked games yet")
        
        with col2:
            st.markdown("### 🎭 Preferred Genres")
            if st.session_state.user_preferences['preferred_genres']:
                for genre in st.session_state.user_preferences['preferred_genres']:
                    st.write(f"• {genre}")
            else:
                st.info("No preferred genres yet")
            
            st.markdown("### 🎮 Preferred Platforms")
            if st.session_state.user_preferences['preferred_platforms']:
                for platform in st.session_state.user_preferences['preferred_platforms']:
                    st.write(f"• {platform}")
            else:
                st.info("No preferred platforms yet")
        
        st.markdown("---")
        
        # Reset preferences
        if st.button("🔄 Reset All Preferences", type="secondary"):
            st.session_state.user_preferences = {
                'liked_games': [],
                'disliked_games': [],
                'preferred_genres': [],
                'preferred_platforms': []
            }
            st.success("✅ Preferences reset!")
            st.rerun()
    
    # Footer
    st.markdown("---")
    st.markdown("""
        <div style='text-align: center; color: #666;'>
            <p>🎮 GameFinder AI - Powered by Machine Learning</p>
            <p>Built with Streamlit, Scikit-learn, and Plotly</p>
        </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()