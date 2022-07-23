import pickle5 as pickle
import numpy as np
import tweepy
import configparser
import pandas as pd
import re
import string

# sklearn TfidfVectorizer libraries
from sklearn.feature_extraction.text import TfidfVectorizer

from flask import render_template, Flask

# read configs
config = configparser.ConfigParser()
config.read('config.ini')

# Global Parameters
api_key = config['twitter']['api_key']
api_key_secret = config['twitter']['api_key_secret']

access_token = config['twitter']['access_token']
access_token_secret = config['twitter']['access_token_secret']

# creating a new dataframe
df = pd.DataFrame(columns=["target","t_id", "created_at", "user", "text"])

# defining a function to retrieve new tweets from API call
def get_tweets(api , topic, count):
    i = 0
    for tweet in tweepy.Cursor(api.search_tweets, q=topic, count=200, lang="en").items():
        df.loc[i, "t_id"] = tweet.id
        df.loc[i, "created_at"] = tweet.created_at
        df.loc[i, "user"] = tweet.user.name
        df.loc[i, "text"] = tweet.text

        i = i + 1
        if i > count:
            break
        else:
            pass

def remove_unwanted_cols(df, cols):
    for col in cols:
        del df[col]
    return df

# We only want the Text so :

# (@[A-Za-z0-9]+)   : Delete Anything like @hello @Letsupgrade etc
# ([^0-9A-Za-z \t]) : Delete everything other than text,number,space,tabspace
# (\w+:\/\/\S+)     : Delete https://
# ([RT]) : Remove "RT" from the tweet

def preprocess_tweet_text(tweet):
    # Convert the text to lowercase
    tweet.lower()

    # Remove urls
    tweet = re.sub(r"http\S+|www\S+|https\S+", '', tweet, flags=re.MULTILINE)

    # Remove user @ references and '#' from tweet
    tweet = re.sub(r'\@\w+|\#','', tweet)

    # Remove punctuations
    tweet = re.sub('(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)|([RT])', ' ', tweet)
    
    tweet_tokens = re.findall("[\w']+", tweet)
    
    return " ".join(tweet_tokens)

def get_feature_vector(train_fit):
    vector = TfidfVectorizer(max_features=500)
    vector.fit(train_fit)
    return vector

##create an instance of flask class for our app.
app = Flask(__name__)

#creates url
@app.route('/')

##Function to create first welcome page of TT.
def table():
    return df.to_html()

##run the application on local deployment server.
if __name__ == "__main__":

    # authentication
    auth = tweepy.OAuthHandler(api_key, api_key_secret)
    auth.set_access_token(access_token, access_token_secret)

    api = tweepy.API(auth)
    public_tweets = api.home_timeline()

    # Call the function to extract the data. pass the topic and filename you want the data to be stored in.
    Topic = ["Ukraine"]
    get_tweets(api, Topic, count=100)

    # Remove unwanted columns from dataset
    df = remove_unwanted_cols(df, ['t_id', 'created_at', 'user'])

    #Preprocess data
    df.text = df['text'].apply(preprocess_tweet_text)

    # Vectorize the input text data
    tf_vector = get_feature_vector(np.array(df.iloc[:, 1]).ravel())

    # Create input variable to pass to the model
    X = tf_vector.transform(np.array(df.iloc[:, 1]).ravel())

    # load the model from disk
    with open('model.pkl' , 'rb') as f:   
        lr = pickle.load(f)
    df['target'] = lr.predict(X)

    print(df.head(50))

    app.run(host='0.0.0.0')
