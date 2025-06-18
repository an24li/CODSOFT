This project constructs a machine learning model to predict movie genres from their plot summaries by employing Natural Language Processing (NLP) methods. It applies TF-IDF vectorization and a Logistic Regression classifier to anticipate genres from textual descriptions.
----------Problem Statement-----------
Predict a movie's genre (e.g., Action, Drama, Comedy) from a textual description of its plot. This can be applied to automatic tagging for streaming websites or film repositories.

This project uses the [IMDB Genre Classification Dataset](https://www.kaggle.com/datasets/hijest/genre-classification-dataset-imdb) from Kaggle.
---------Dataset-----------
Dataset used:📥 Genre Classification Dataset - IMDb
Columns:
Text: The plot summary of the movie
Genre: The actual genre label (target class)
Please download the dataset and save it as Genre Classification Dataset.csv in the root directory before running the code.

-----Approach------
1.Data Preprocessing
a.Lowercasing text
b.Handling missing values

2.Feature Extraction
a.TF-IDF Vectorization (max_features=5000)

3.Model
a.Logistic Regression with max_iter=1000 for better convergence

4.Evaluation
a.Accuracy Score
b.Classification Report (Precision, Recall, F1-Score)

---------How to Run-----------

a.Clone this repository or download the files.
b.Download the dataset from Kaggle and put it in the same folder.
c.Install the necessary libraries:
pip install -r requirements.txt
