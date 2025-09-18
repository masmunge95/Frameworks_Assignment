# Streamlit App
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

@st.cache_data
def load_data():
    """
    Loads and preprocesses the dataset.
    - Uses Streamlit's caching to run only once, making the app much faster.
    - Converts 'publish_time' to datetime and extracts the year.
    """
    try:
        df = pd.read_csv('data/cleaned_metadata.csv')
        # Convert 'publish_time' to datetime, coercing errors to NaT (Not a Time)
        df['publish_time'] = pd.to_datetime(df['publish_time'], errors='coerce')
        # Drop rows where date conversion failed
        df.dropna(subset=['publish_time'], inplace=True)
        # Extract year and convert to integer
        df['publish_year'] = df['publish_time'].dt.year.astype(int)
        return df
    except FileNotFoundError:
        st.error(
            "⚠️ 'cleaned_metadata.csv' not found in the 'data/' folder.\n\n"
            "👉 Please make sure you:\n"
            "1. Download the raw `metadata.csv` from Kaggle.\n"
            "2. Place it inside the `data/` folder in the project root.\n"
            "3. Run the Jupyter Notebook (`notebooks/CORD-19_Analysis.ipynb`) to generate `cleaned_metadata.csv`.\n\n"
            "See the README.md for detailed instructions."
        )
        return None

# --- Streamlit App Setup ---

st.title("CORD-19 Dataset Analysis")

# Load data with a spinner for user feedback during the initial (slow) load
with st.spinner('Loading and processing data... This may take a moment on first run.'):
    df = load_data()

# --- Main App Logic ---
# Only run the rest of the app if the data was loaded successfully
if df is not None:
    st.write("This app provides an analysis of the CORD-19 dataset, including data cleaning, basic statistics, and visualizations.")

    # ===== Sidebar Filters =====
    st.sidebar.header("Filters")

    # Define min/max years for sliders
    min_year, max_year = int(df['publish_year'].min()), int(df['publish_year'].max())

    # Initialize session state for filters to remember user's choices
    if "year_range" not in st.session_state:
        st.session_state.year_range = (min_year, max_year)
    if "selected_journal" not in st.session_state:
        st.session_state.selected_journal = "All"
    if "selected_source" not in st.session_state:
        st.session_state.selected_source = "All"

    # Reset button
    if st.sidebar.button("Reset Filters"):
        st.session_state.year_range = (min_year, max_year)
        st.session_state.selected_journal = "All"
        st.session_state.selected_source = "All"
        st.rerun()

    # Year filter
    year_filter = st.sidebar.slider(
        "Select Year Range", min_year, max_year, value=st.session_state.year_range
    )
    st.session_state.year_range = year_filter

    # Journal filter
    journal_options = ["All"] + sorted(df['journal'].dropna().unique().tolist())
    selected_journal = st.sidebar.selectbox(
        "Select Journal", options=journal_options, index=journal_options.index(st.session_state.selected_journal)
    )
    st.session_state.selected_journal = selected_journal

    # Source filter
    source_options = ["All"] + sorted(df['source_x'].dropna().unique().tolist())
    selected_source = st.sidebar.selectbox(
        "Select Source", options=source_options, index=source_options.index(st.session_state.selected_source)
    )
    st.session_state.selected_source = selected_source

    # --- Streamlined Filtering Logic ---
    # Apply year and source filters first. This is used for the Top Journals chart.
    df_main_filtered = df[
        (df['publish_year'] >= year_filter[0]) &
        (df['publish_year'] <= year_filter[1])
    ]
    if selected_source != "All":
        df_main_filtered = df_main_filtered[df_main_filtered['source_x'] == selected_source]

    # Now, apply the journal filter to create the final filtered df for all other charts.
    df_filtered = df_main_filtered
    if selected_journal != "All":
        df_filtered = df_filtered[df_filtered['journal'] == selected_journal]

    st.metric("Total Articles in Selection", f"{len(df_filtered):,}")

    # ===== Articles Published Over Time =====
    st.subheader("Articles Published Over Time")
    if not df_filtered.empty:
        df_temp = df_filtered.groupby('publish_year').size().reset_index(name='Articles')
        df_temp = df_temp.sort_values('publish_year')

        fig, ax = plt.subplots(figsize=(10, 6))
        sns.lineplot(data=df_temp, x='publish_year', y='Articles', marker='o', ax=ax)
        ax.set_title('Articles Published Over Time')
        ax.set_xlabel('Year')
        ax.set_ylabel('Number of Articles')
        ax.grid(True)
        st.pyplot(fig)
    else:
        st.write("No articles to display for the current filter selection.")

    # ===== Top 10 Journals =====
    st.subheader("Top 10 Journals by Article Count")
    if not df_main_filtered.empty:
        bar_temp = df_main_filtered['journal'].value_counts().head(10).reset_index()
        bar_temp.columns = ['Journal', 'Articles']

        fig2, ax2 = plt.subplots(figsize=(10, 8))
        sns.barplot(data=bar_temp, x='Articles', y='Journal', palette='viridis', ax=ax2)
        ax2.set_title('Top 10 Journals by Article Count')
        ax2.set_xlabel('Number of Articles')
        ax2.set_ylabel('Journal')
        plt.tight_layout()
        st.pyplot(fig2)
    else:
        st.write("No journal data to display for the current filter selection.")

    # ===== Word Cloud =====
    st.subheader("Word Cloud of Most Common Words in Titles")
    if not df_filtered.empty:
        text = " ".join(title for title in df_filtered['title'].dropna() if title)
        if text:
            wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
            fig3, ax3 = plt.subplots(figsize=(10, 6))
            ax3.imshow(wordcloud, interpolation='bilinear')
            ax3.axis('off')
            ax3.set_title('Word Cloud of Most Common Words in Titles')
            st.pyplot(fig3)
        else:
            st.write("No titles available to generate a word cloud.")
    else:
        st.write("No titles available to generate a word cloud.")

    # ===== Distribution of Article Counts by Source =====
    st.subheader("Distribution of Article Counts by Source")
    if not df_filtered.empty:
        source_temp = df_filtered['source_x'].value_counts().reset_index()
        source_temp.columns = ['Source', 'Articles']

        fig4, ax4 = plt.subplots(figsize=(12, 10))
        sns.barplot(data=source_temp, x='Source', y='Articles', palette='viridis', ax=ax4)
        ax4.set_title('Distribution of Article Counts by Source')
        ax4.set_xlabel('Source')
        ax4.set_ylabel('Number of Articles')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()

        # Annotate bars
        for p in ax4.patches:
            ax4.annotate(int(p.get_height()), (p.get_x() + p.get_width()/2., p.get_height()),
                         ha='center', va='bottom')
        st.pyplot(fig4)
    else:
        st.write("No source data to display for the current filter selection.")
