# Importing necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
def load_and_prepare_data():
    """Load and prepare the SMS spam dataset"""
    # Loading the dataset (using latin-1 encoding to handle special characters)
    data = pd.read_csv("spam.csv", encoding='latin-1')
    
    # Selecting only the relevant columns and renaming them
    data = data[['v1', 'v2']]
    data.columns = ['label', 'message']
    
    return data

def explore_data(df):
    """Perform basic data exploration"""
    print("\nFirst 5 rows of the dataset:")
    print(df.head())
    
    print("\nLabel distribution:")
    print(df['label'].value_counts())
    plt.figure(figsize=(6, 4))
    sns.countplot(x='label', data=df)
    plt.title('Distribution of Spam vs Ham Messages')
    plt.xlabel('Label (0=Ham, 1=Spam)')
    plt.ylabel('Count')
    plt.show()

def preprocess_data(df):
    """Convert labels to numerical values and split data"""
    df['label_num'] = df['label'].map({'ham': 0, 'spam': 1})
    X = df['message']
    y = df['label_num']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)
    
    return X_train, X_test, y_train, y_test

def vectorize_text(X_train, X_test):
    """Convert text messages to TF-IDF features""" 
   tfidf = TfidfVectorizer(stop_words='english', max_df=0.9)
    
    X_train_tfidf = tfidf.fit_transform(X_train)
    X_test_tfidf = tfidf.transform(X_test)
    
    return X_train_tfidf, X_test_tfidf, tfidf

def train_and_evaluate(X_train_tfidf, y_train, X_test_tfidf, y_test):
    """Train a Naive Bayes model and evaluate performance""
    nb_model = MultinomialNB()
    nb_model.fit(X_train_tfidf, y_train)
    y_pred = nb_model.predict(X_test_tfidf)
    
  
    print("\nModel Evaluation:")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Ham', 'Spam'], 
                yticklabels=['Ham', 'Spam'])
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()
    
    return nb_model

def main():
    df = load_and_prepare_data()
    explore_data(df)
    X_train, X_test, y_train, y_test = preprocess_data(df)
    X_train_tfidf, X_test_tfidf, tfidf = vectorize_text(X_train, X_test)
    model = train_and_evaluate(X_train_tfidf, y_train, X_test_tfidf, y_test)

if __name__ == "__main__":
    main()
