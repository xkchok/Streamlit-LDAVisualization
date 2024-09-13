## Visualization of Latent Dirichlet Allocation (LDA) Topic Modeling on Netflix Shows Dataset

Streamlit URL: https://app-ldavisualization-hyajjpblvw3gnl8xhzz3p4.streamlit.app/

---

This application performs topic modeling on the 'Description' column from the [Netflix Shows dataset](https://www.kaggle.com/datasets/shivamb/netflix-shows/data). The dataset contains listings of movies and TV shows available on Netflix.

### Dataset Information
- **Source**: Kaggle
- **Columns**: The dataset includes various columns such as `show_id`, `type`, `title`, `director`, `cast`, `country`, `date_added`, `release_year`, `rating`, `duration`, `listed_in`, and `description`.
- **Description Column**: The 'Description' column provides a brief summary of each movie or TV show.

### Purpose
The purpose of this application is to use Latent Dirichlet Allocation (LDA) to identify topics within the descriptions of Netflix shows. The visualization below shows the topics and their relationships.

### Model Parameters
For this topic modeling, the maximum number of features for the TF-IDF vectorizer is set to 500, and the number of topics (components) for the LDA model is set to 3.