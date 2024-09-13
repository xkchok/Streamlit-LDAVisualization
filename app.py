import streamlit as st
import pyLDAvis
import pyLDAvis.lda_model
import joblib

# Load the saved model and data
best_lda_model = joblib.load('best_lda_model.pkl')
X_transformed = joblib.load('X_transformed.pkl')
best_vectorizer = joblib.load('best_vectorizer.pkl')

# Prepare the LDA visualization
panel = pyLDAvis.lda_model.prepare(best_lda_model, X_transformed, best_vectorizer, mds='tsne')

# Convert the visualization to HTML
html_string = pyLDAvis.prepared_data_to_html(panel)

# Streamlit app
st.title("LDA Visualization with pyLDAvis")

# Add context about the dataset and topic modeling
st.markdown("""
## Topic Modeling on Netflix Shows Dataset

This application performs topic modeling on the 'Description' column from the [Netflix Shows dataset](https://www.kaggle.com/datasets/shivamb/netflix-shows/data). The dataset contains listings of movies and TV shows available on Netflix.

### Dataset Information
- **Source**: Kaggle
- **Columns**: The dataset includes various columns such as `show_id`, `type`, `title`, `director`, `cast`, `country`, `date_added`, `release_year`, `rating`, `duration`, `listed_in`, and `description`.
- **Description Column**: The 'Description' column provides a brief summary of each movie or TV show.

### Purpose
The purpose of this application is to use Latent Dirichlet Allocation (LDA) to identify topics within the descriptions of Netflix shows. The visualization below shows the topics and their relationships.

### Model Parameters
For this topic modeling, the maximum number of features for the TF-IDF vectorizer is set to 500, and the number of topics (components) for the LDA model is set to 3.

""")

# Display the HTML in Streamlit
st.components.v1.html(html_string, width=1300, height=800)