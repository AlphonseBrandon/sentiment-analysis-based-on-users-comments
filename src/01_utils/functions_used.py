'''
Author: Alphonse Brandon
Last Updated Date: 11/17/2022
Last Updated Time: 12:30 AM

Description: This script contains all the function used in building an nlp model based on users ratings and comments
'''

import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn import naive_bayes
import pickle

def download_stopwords():
    '''This function downloads the already cathegorized words from the natural language toolkit (nlt)'''
    nltk.download('stopwords')

def initialising_paths():
    '''This function points where data for modeling is located and where to save the model after modeling'''

    global path_to_data
    global save_the_mode_path
    save_the_mode_path = 'D:/github-repos/nlp-sentiment-analysis/models/nlp_model.pkl'
    path_to_data = 'D:/github-repos/nlp-sentiment-analysis/data/01_raw/reviews.txt'

def load_data():
    '''This function loads the model from the reviews text file and sets them into a two column dataframe'''
    global reviews_data
    reviews_data = pd.read_csv(path_to_data, sep='\t', names=['Reviews', 'Comments'])

def set_language():
    '''Get the english version of the cathegorized text from nltk'''
    global stopset
    stopset = set(stopwords.words('english'))

def make_vectorizer():
    '''Initialising the function to turn text into numbers so the computer can understand'''
    global vectorizer
    vectorizer = TfidfVectorizer(use_idf=True, lowercase=True, strip_accents='ascii', stop_words=stopset)

def load_the_features():
    '''This function converts the review text into numbers'''
    global X, y
    X = vectorizer.fit_transform(reviews_data.Comments)
    y = reviews_data.Reviews
    pickle.dump(vectorizer, open('transform', 'wb'))

def split_the_data():
    '''This funstion set the foundation to train and test the accuracy of the model'''
    global X_train, X_test, y_train, y_test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

def teach_the_model():
    '''This function fits the data to the model for training'''
    global clf
    clf = naive_bayes.MultinomialNB()
    clf.fit(X_train, y_train)

def get_the_model_accuracy():
    '''This function shows us how well our model will perform on unseen data'''
    pred = clf.predict(X_test)
    return accuracy_score(y_test, pred)*100

def save_the_mode():
    '''This function saves the trained model'''
    pickle.dump(clf, open(save_the_mode_path, 'wb'))

def get_first_five_sentiments():
    '''This function gets the sentiments from the comments and categorizes them as good or bad'''
    reviews_list = []
    reviews_status = []

    for reviews in reviews_data['Comments'][0:5]:
        if reviews:
            reviews_list.append(reviews)
            '''adding the comments from the reviews text into the model'''
            movie_review_list = np.array([reviews])
            movie_vector = vectorizer.transform(movie_review_list)
            pred = clf.predict(movie_vector)
            reviews_status.append('Good' if pred else 'Bad')

    # Combining reviews and comments into a dictionary
    movie_reviews = {reviews_list[i]: reviews_status[i] for i in range(len(reviews_list))}
    return movie_reviews
