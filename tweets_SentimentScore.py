from fileinput import filename
import numpy as np
import tweepy
import configparser
import pandas as pd
import re
from textblob import TextBlob 

# NLP preprocessing libraries
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer

# read configs
config = configparser.ConfigParser()
config.read('config.ini')


api_key = config['twitter']['api_key']
api_key_secret = config['twitter']['api_key_secret']

access_token = config['twitter']['access_token']
access_token_secret = config['twitter']['access_token_secret']


# authentication
auth = tweepy.OAuthHandler(api_key, api_key_secret)
auth.set_access_token(access_token, access_token_secret)

api = tweepy.API(auth)

public_tweets = api.home_timeline()

df = pd.DataFrame(columns=["target","t_id", "created_at", "user", "text"])
print(df)


def get_tweets(topic, count):
    i = 0
    for tweet in tweepy.Cursor(api.search_tweets, q=topic, count=100, lang="en").items():
        #print(i, end='\r')
        df.loc[i, "t_id"] = tweet.id
        df.loc[i, "created_at"] = tweet.created_at
        #df.loc[i, "query"] = tweet.query
        df.loc[i, "user"] = tweet.user.name
        #df.loc[i, "IsVerified"] = tweet.user.verified
        df.loc[i, "text"] = tweet.text
        #df.loc[i, "likes"] = tweet.favorite_count
        df.loc[i, "retweet"] = tweet.retweet_count
        #df.loc[i, "User_location"] = tweet.user.location
        df.to_csv('data/sample.csv')
        i = i + 1
        if i > count:
            break
        else:
            pass


# Call the function to extract the data. pass the topic and filename you want the data to be stored in.
Topic = ["Ukraine"]

get_tweets(Topic, count=500)
#df= pd.read_csv('data/trail_target_allpolarities.csv')

def clean_tweet(tweet):
    return ' '.join(re.sub('(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)|([RT])', ' ', str(tweet).lower()).split())



def analyze_sentiment(tweet):
    analysis = TextBlob(tweet)
    print(analysis.sentiment.polarity)
    if analysis.sentiment.polarity in np.arange(-1,-0.5):
        return 'Highly Negative'
    elif analysis.sentiment.polarity in np.arange(-0.5,0):
        return 'Slightly Negative'
    elif analysis.sentiment.polarity in np.arange(0,0.5):
        return 'Slightly Positive'
    elif analysis.sentiment.polarity in np.arange(0.5,1):
        return 'Positive'
    else:
        return 'Neutral'


   

df['text'] = df['text'].apply(lambda x: clean_tweet(x))

print(df.head(5))

df["target"] = df["text"].apply(lambda x: analyze_sentiment(x))

print(df.head(50))


print("Unique values of target : " , df.target.unique())

print("value count:" ,df['target'].value_counts())

df.to_csv('data/sample_target.csv')
