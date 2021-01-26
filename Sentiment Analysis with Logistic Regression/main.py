from nltk.corpus import twitter_samples
from model import train_model, predict_tweet,  test_model
from utils import build_freqs
import numpy as np
all_positive_tweets = twitter_samples.strings('positive_tweets.json') # num:5000; type:list
all_negative_tweets = twitter_samples.strings('negative_tweets.json') # num:5000; type:list
test_pos = all_positive_tweets[4000:]
train_pos = all_positive_tweets[:4000]
test_neg = all_negative_tweets[4000:]
train_neg = all_negative_tweets[:4000]
train_x = train_pos + train_neg 
test_x = test_pos + test_neg

train_y = np.append(np.ones((len(train_pos), 1)), np.zeros((len(train_neg), 1)), axis=0)
test_y = np.append(np.ones((len(test_pos), 1)), np.zeros((len(test_neg), 1)), axis=0)

freqs = build_freqs(train_x, train_y)
theta = train_model(train_x,train_y,freqs)
accuracy = test_model(test_x, test_y, freqs, theta)
print("The accuracy of sigmoid model for tweets sentimental classification is: " + accuracy)





