from distutils.command.clean import clean
from Tweet_prediction import Tweet_prediction
import twint
import pandas as pd
import re
from Model import Model

class HSInspector():
    def __init__(self) -> None:
        self.tweet_predictions = Tweet_prediction()
        self.model = Model()
        self.model.load_model('./trained_model/xlnet_model4.bin')

    def searchTweets(self, keyword):
        df = self.scrapeTweetsByKeyword(keyword=keyword)
        self.tweet_predictions.get_tweets(df)
        self.clean_tweets()
        types = []
        for i in range(self.tweet_predictions.size):
            types.append(self.model.predict_tweet(self.tweet_predictions.tweets[i].tweet))
        self.tweet_predictions.updatePredictions(types)
        js = self.tweet_predictions.to_json(username=df['username'].tolist())
        # js.update({'username': df['username'].tolist()})
        return js
        # print(js)
        # self.tweet_predictions.print_tweets()

    def searchTweetsByUsername(self, username):
        df = self.scrapeTweetsByUseraname(username=username)
        self.tweet_predictions.get_tweets(df)
        self.clean_tweets()
        types = []
        for i in range(self.tweet_predictions.size):
            types.append(self.model.predict_tweet(self.tweet_predictions.tweets[i].tweet))
        self.tweet_predictions.updatePredictions(types)
        js = self.tweet_predictions.to_json(username=df['username'].tolist())
        return js


    def scrapeTweetsByKeyword(self, keyword="Hello"):
        self.c = twint.Config()
        self.c.Lang = "english"
        self.c.Limit= 10
        self.c.Hide_output = True
        self.c.Pandas=True
        
        self.c.Search = keyword
        v = twint.run.Search(self.c)
        return twint.storage.panda.Tweets_df

    def scrapeTweetsByUseraname(self, username):
        self.c = twint.Config()
        self.c.Lang = "english"
        self.c.Limit= 10
        self.c.Hide_output = True
        self.c.Pandas=True
        self.c.Username = username
        v = twint.run.Search(self.c)
        return twint.storage.panda.Tweets_df

    def clean_tweets(self):
        cleaned_tweets = []
        for i in range(self.tweet_predictions.size):
            cleaned_tweets.append(self.clean_text(self.tweet_predictions.tweets[i].tweet))
        self.tweet_predictions.updateCleanedTweets(cleaned_tweets)

    def clean_text(self, text):
        text = re.sub("RT",' ',text)
        text = re.sub("!{3,}",' ',text)
        text = re.sub(r"&#8217;", "'", text)
        text = re.sub(r"&[#*a-zA-z0-9]+;", '', text)
        text = re.sub("@[_*a-zA-Z0-9]+:*",' ',text)
        text = re.sub(r"https?://[A-Za-z0-9./]+", ' ', text)
        text = re.sub('\t+', '',  text)
        text = re.sub(" {2,}",' ',text)
        text = re.sub("^ ",'',text)
        text = text.lower()
        return text



