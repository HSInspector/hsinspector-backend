def func(text):
    print("from test file")
    print(text)

import twint
import pandas as pd

c = twint.Config()
c.Search = "fruits"
c.Lang = "en"
c.Limit= 10
c.Pandas = True
c.Hide_output = True

# Run
twint.run.Search(c)
test_df = twint.storage.panda.Tweets_df
tweets = test_df['tweet']
print(tweets)