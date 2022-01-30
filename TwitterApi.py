# This is a sentiment analysis program that parses tweets fetched from twitter

# import libraries

import tweepy
from textblob import TextBlob
from wordcloud import WordCloud
import pandas as pd
import numpy as np
import re
import TwitterAPIKeys
import matplotlib.pyplot as plt

consumerKey = TwitterAPIKeys.CONSUMER_KEY
consumerSecret = TwitterAPIKeys.CONSUMER_KEY_SECRET
accessToken = TwitterAPIKeys.ACCESS_TOKEN
accessTokenSecret = TwitterAPIKeys.ACCESS_TOKEN_SECRET

plt.style.use('fivethirtyeight')

# create authentication object
authenticate = tweepy.OAuthHandler(consumerKey, consumerSecret)

#Set the access token and secret
authenticate.set_access_token(accessToken, accessTokenSecret)

#Create the api object while passing in the authenticator
api = tweepy.API(authenticate, wait_on_rate_limit=True)

#extract tweets from user
posts = api.user_timeline(screen_name = "BusinessInsider", count=100, tweet_mode="extended")

#create dataframe to contain the tweets
df = pd.DataFrame([tweet.full_text for tweet in posts], columns=["Tweets"])

#Clean the data
def cleanTxt(text):
    text = re.sub(r'@[A-Za-z0-9]+', '', text) #remove @mention
    text = re.sub(r'#', '', text) #remove #
    text = re.sub(r'RT[\s]+', '', text)
    text = re.sub(r'https?:\/\/\S+', '', text)

    return text

#cleaning the text
df['Tweets'] = df['Tweets'].apply(cleanTxt)

#show the cleaned txt
print(df)


#Create Function to get subjectivity
def getSubjectivity(text):
    return TextBlob(text).sentiment.subjectivity

#Create function to return polarity
def getPolarity(text):
    return TextBlob(text).sentiment.polarity

#add subjectivity and polarity to df
df['Subjectivity'] = df['Tweets'].apply(getSubjectivity)
df['Polarity'] = df['Tweets'].apply(getPolarity)

#plot wordCloud
allWords = ' '.join([twts for twts in df['Tweets']])
WordCloud = WordCloud(width = 500, height = 300, random_state = 21, max_font_size = 110).generate(allWords)

plt.imshow(WordCloud, interpolation = "bilinear")
plt.axis('off')
plt.show()

def getAnalysis(score):
    if score < 0:
        return 'Negative'
    elif score == 0:
        return 'Neutral'
    else:
        return 'Positive'

df['Analysis'] = df['Polarity'].apply(getAnalysis)

#print all of the positive tweets
j = 1
sortedDF = df.sort_values(by=['Polarity'], ascending=False)
sortedDF = sortedDF.reset_index(drop=True)
for i in range(sortedDF.shape[0], 0):
    if sortedDF['Analysis'] == 'Positive':
        print(str(j) + ') ' + sortedDF['Tweets'][i])
        j += 1
        
plt.figure(figsize=(8,6))
for i in range(df.shape[0]):
    plt.scatter(df['Polarity'][i], df['Subjectivity'][i])
plt.title('Sentiment Analysis')
plt.xlabel('Polarity')
plt.ylabel('Subjectivity')
plt.show()
     

