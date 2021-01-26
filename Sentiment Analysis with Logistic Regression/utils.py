'''
utils: 
    preprocess_tweet(tweet)                    # Preprocess single tweet
    build_freqs()                              # Build word frequencies for all tweets 
'''
import nltk                                # Python library for NLP
from nltk.corpus import twitter_samples    # sample Twitter dataset from NLTK
import re                                  # library for regular expression operations
import string                              # for string operations
from nltk.corpus import stopwords          # module for stop words that come with NLTK
from nltk.stem import PorterStemmer        # module for stemming
from nltk.tokenize import TweetTokenizer   # module for tokenizing strings
import numpy as np                         # library for scientific computing and matrix operations
nltk.download('twitter_samples')
nltk.download('stopwords')

def preprocess_tweet(tweet):
    '''
    Input: single tweet
    Ouput: processed tweet
    
    '''
    tweets_clean = []
    tweets_stem = []
    stemmer = PorterStemmer()
    # 消除特殊符号
    tweet = re.sub(r'@[\s]+', '', tweet)
    tweet = re.sub(r'https?:\/\/.*[\r\n]*', '', tweet)
    tweet = re.sub(r'#', '', tweet)
    # 分词
    tokenizer = TweetTokenizer(preserve_case=False, strip_handles=True, reduce_len=True)
    tweet_tokens = tokenizer.tokenize(tweet)
    # 消除停止词和标点符号
    stopwords_english = stopwords.words('english')
    for word in tweet_tokens:
        if(word not in stopwords_english and word not in string.punctuation):
            # stemming
            stem_word = stemmer.stem(word)
            tweets_clean.append(stem_word)
    return tweets_clean

def build_freqs(tweets, ys):
    '''
    Input:
        tweets: a list of tweets
        ys: an m x 1 array with the sentiment label of each tweet (either 0 or 1)
    Output:
        freqs: a dictionary mapping each (word, sentiment) pair to its frequency
    '''
    freqs = {}
    yslist = np.squeeze(ys).tolist()
    for y,tweet in zip(yslist, tweets):
        for word in preprocess_tweet(tweet):
            pair = (word, y)
            freqs[pair] = freqs.get(pair, 0) + 1
    return freqs






    

  
