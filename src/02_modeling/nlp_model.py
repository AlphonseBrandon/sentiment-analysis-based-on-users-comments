'''
Author: Alphonse Brandon
Last Updated Date: 11/17/2022
Last Updated Time: 12:34 AM

Description: Run this script to get the sentiment analysis on a comment and review in our reviews.txt file for the first five comments
'''

import sys
sys.path.insert(1, 'D:/github-repos/nlp-sentiment-analysis/src/01_utils')
import functions_used

import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn import naive_bayes
import pickle


functions_used.preprocessing()