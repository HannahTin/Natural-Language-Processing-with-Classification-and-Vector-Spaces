import nltk
from os import getcwd
import numpy as np
import pandas as pd
from nltk.corpus import twitter_samples 
from utils import preprocess_tweet, build_freqs

def extract_features(tweet, freqs):
    '''
    Input: 
        tweet: a list of words for one tweet
        freqs: a dictionary corresponding to the frequencies of each tuple (word, label)
    Output: 
        x: a feature vector of dimension (1,3)
    '''
    word_list = preprocess_tweet(tweet)
    x = np.zeros((1,3))
    x[0,0] = 1
    for word in word_list:
        x[0,1] += freqs.get((word,1),0)
        x[0,2] += freqs.get((word,0),0)
    assert(x.shape == (1,3))
    return x

def sigmoid(z):
    '''
    Input:
        z: is the input (can be a scalar or an array)
    Output:
        h: the sigmoid of z
    '''
    h = 1/(1+np.exp(-z))
    return h

def gradientDesecent(x, y, theta, alpha, num_iters):
    '''
    Input:
        x: matrix of features which is (m,n+1)
        y: corresponding labels of the input matrix x, dimensions (m,1)
        theta: weight vector of dimension (n+1,1)
        alpha: learning rate
        num_iters: number of iterations you want to train your model for
    Output:
        J: the final cost
        theta: your final weight vector
    Hint: you might want to print the cost to make sure that it is going down.
    '''
    m = x.shape[0]
    for i in range(num_iters):
        z = np.dot(x, theta)
        h = sigmoid(z)
        J = -1./ m * (np.dot(y.T,np.log(h)) + np.dot((1-y).T, np.log(1-h)))
        theta = theta - alpha /m * np.dot(x.T,(h-y))
    J = float(J)
    return J, theta

def train_model(train_x,train_y,freqs):
   X = np.zeros((len(train_x), 3))
   for i in range(len(train_x)):
       X[i, :]= extract_features(train_x[i], freqs)
       Y = train_y
   J, theta = gradientDesecent(X, Y, np.zeros((3, 1)), 1e-9, 1500)
   return theta

def predict_tweet(tweet, freqs, theta):
    '''
    Input: 
        tweet: a string
        freqs: a dictionary corresponding to the frequencies of each tuple (word, label)
        theta: (3,1) vector of weights
    Output: 
        y_pred: the probability of a tweet being positive or negative
    '''
    x = extract_features(tweet, freqs)
    y_pred = sigmoid(np.dot(x,theta))
    return y_pred


def test_model(test_x, test_y, freqs, theta):
    """
    Input: 
        test_x: a list of tweets
        test_y: (m, 1) vector with the corresponding labels for the list of tweets
        freqs: a dictionary with the frequency of each pair (or tuple)
        theta: weight vector of dimension (3, 1)
    Output: 
        accuracy: (# of tweets classified correctly) / (total # of tweets)
    """
    y_hat = []
    for tweet in test_x:
        y_pred = predict_tweet(tweet, freqs, theta)
        if y_pred > 0.5:
            y_hat.append(1.0)
        else:
            y_hat.append(0)
    # With the above implementation, y_hat is a list, but test_y is (m,1) array
    # convert both to one-dimensional arrays in order to compare them using the '==' operator
    accuracy = np.sum(np.asarray(y_hat)==np.squeeze(test_y))/len(y_hat)
    return accuracy










