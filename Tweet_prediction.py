from Tweet import Tweet
from hate_speech_type import hate_speech_type
import json

class Tweet_prediction():
    def __init__(self) -> None:
        self.tweets = []
        self.size = 0

    def updateCleanedTweets(self, tweets):
        for i in range(len(self.tweets)):
            self.tweets[i].set_tweet(tweets[i])
    
    def updatePredictions(self, preds):
        for i in range(len(self.tweets)):
            self.tweets[i].set_type(preds[i])

    def get_tweets(self, df):
        self.tweets = []
        tweets = df['tweet']
        for i in range(len(df)):
            temp = Tweet(tweets[i], hate_speech_type.none)
            self.tweets.append(temp)
        self.size = len(self.tweets)
        return self.tweets

    def print_tweets(self):
        for i in range(len(self.tweets)):
            print("tweet: ", self.tweets[i].tweet, " predicted: ", self.tweets[i].type)

    def to_json(self, **extra_params):
        tweets = []
        types = []
        for i in range(self.size):
            tweets.append(self.tweets[i].tweet)
            types.append(self.tweets[i].type.value)
 
        x = {'tweets':tweets, 'types': types}
        for key, value in extra_params.items():
            x.update({key:value})

        return json.dumps(x)