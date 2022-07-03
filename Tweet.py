from hate_speech_type import hate_speech_type

class Tweet():
    def __init__(self, tweet: str="",type: hate_speech_type=hate_speech_type.none ) -> None:
        self.tweet = tweet
        self.type = type
    
    def set_tweet(self, tweet: str):
        self.tweet = tweet

    def set_type(self, type: hate_speech_type):
        self.type = type
