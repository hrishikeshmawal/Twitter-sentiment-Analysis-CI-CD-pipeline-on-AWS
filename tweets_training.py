import pandas as pd
import numpy as np
import re
import string
import sys

# NLP preprocessing libraries
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
#from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer

# ML Libraries
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

# MLFlow libraries
from urllib.parse import urlparse
import mlflow
import mlflow.sklearn
# Logger library
import logging

logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)

# Global Parameters
stop_words = set(stopwords.words('english'))

max_features = sys.argv[1] if len(sys.argv) > 1 else 500
#solver = sys.argv[2] if len(sys.argv) > 2 else 'lbfgs'

def load_dataset(filename, cols):
    dataset = pd.read_csv(filename, encoding='latin-1')
    dataset.columns = cols
    return dataset

def remove_unwanted_cols(dataset, cols):
    for col in cols:
        del dataset[col]
    return dataset

def preprocess_tweet_text(tweet):
    # Convert the text to lowercase
    tweet.lower()
    # Remove urls
    tweet = re.sub(r"http\S+|www\S+|https\S+", '', tweet, flags=re.MULTILINE)
    # Remove user @ references and '#' from tweet
    tweet = re.sub(r'\@\w+|\#','', tweet)

    # Remove punctuations
    tweet = re.sub('(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)|([RT])', ' ', tweet)
    
    #tweet = tweet.translate(str.maketrans('', '', string.punctuation))
    
    tweet_tokens = word_tokenize(tweet)
    #tweet_tokens = re.findall("[\w']+", tweet)

    # Remove stopwords
    filtered_words = [w for w in tweet_tokens if not w in stop_words]
    
    #ps = PorterStemmer()
    #stemmed_words = [ps.stem(w) for w in filtered_words]
    lemmatizer = WordNetLemmatizer()
    lemma_words = [lemmatizer.lemmatize(w, pos='a') for w in filtered_words]
    
    return " ".join(lemma_words)

def get_feature_vector(train_fit):
    vector = TfidfVectorizer(max_features=int(max_features))
    vector.fit(train_fit)
    return vector

# Load dataset
dataset = load_dataset("data/trail_allpolarities.csv", ['target', 't_id', 'created_at', 'user', 'text'])
# Remove unwanted columns from dataset
n_dataset = remove_unwanted_cols(dataset, ['t_id', 'created_at', 'user'])
#Preprocess data
dataset.text = dataset['text'].apply(preprocess_tweet_text)

print("dataset : "  , dataset)

print("dataset columns : "  , dataset.columns)

print("Unique values of target : " , dataset.target.unique())
# Split dataset into Train, Test

# Same tf vector will be used for Testing sentiments on unseen trending data
tf_vector = get_feature_vector(np.array(dataset.iloc[:, 1]).ravel())
X = tf_vector.transform(np.array(dataset.iloc[:, 1]).ravel())
y = np.array(dataset.iloc[:, 0]).ravel()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


#print("solver is :", solver)
#mlflow.set_tracking_uri("http://127.0.0.1:5000")
#mlflow.tracking.set_tracking_uri("file:/files/files/model")

with mlflow.start_run():
    # Training Naive Bayes model
    """NB_model = MultinomialNB()
    NB_model.fit(X_train, y_train)
    y_predict_nb = NB_model.predict(X_test)
    NB_accuracy = accuracy_score(y_test, y_predict_nb)
    #print("",accuracy_score(y_test, y_predict_nb))"""

    # Training Logistics Regression model
    """LR_model = LogisticRegression(C=1, penalty='l2', solver=solver)
    LR_model.fit(X_train, y_train)
    y_predict_lr = LR_model.predict(X_test)
    Logistics_accuracy = accuracy_score(y_test, y_predict_lr)
    print("Logistics Regression accoracy : " ,Logistics_accuracy)"""

    # Training Logistics Regression model
    SVC_model = SVC()
    SVC_model.fit(X_train, y_train)
    y_predict_svc = SVC_model.predict(X_test)
    SVC_accuracy = accuracy_score(y_test, y_predict_svc)

    """print("Logistics Regression accoracy : " ,Logistics_accuracy)
    print("Logistics Regression (solver=%s):" % (solver))"""
    print(" accuracy: %s" % SVC_accuracy)

    mlflow.log_param("Feature ", max_features)
    #mlflow.log_param("Solver ", solver)
    mlflow.log_metric("accuracy", SVC_accuracy)

    tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

    # Model registry does not work with file store
    if tracking_url_type_store != "file":

    # Register the model
    # There are other ways to use the Model Registry, which depends on the use case,
    # please refer to the doc for more information:
    # https://mlflow.org/docs/latest/model-registry.html#api-workflow
        mlflow.sklearn.log_model(SVC_model, "model", registered_model_name="SVC")
    else:
        mlflow.sklearn.log_model(SVC_model  , "model")
