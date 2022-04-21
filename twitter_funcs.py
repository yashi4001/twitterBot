# Imports
import os
import numpy as np
import pandas as pd

import pickle
import tweepy

from datetime import datetime,timezone
import re
import time




# twitter_keys = {
#     'consumer_key': '1UNBpZpFcFdkfoUS5AtdEtQHI', #api key
#     'consumer_secret': 'Uq5h1kzAdedJ4Eh67Msv4XlD0LitC4GHjJK1pDJYZuqhoX6Ega', #api key secret
#     'access_token_key': '1492164543220699141-xND4rC2lCyRy6nmwcgvrMAinKgo1eU',
#     'access_token_secret': 'usS0ptKodbgp7DwRfQwcCZhrseN7YpXDRTaE4HIZStVFP'
# }
consumer_key = "ApNzGNFgfROdOG6gniJSBidpT"
consumer_secret = "b4uPxEBhvOoxEpiQjj5xbL2123XXfej2i7LkZM9NjzWCAfeZ7o"
access_key = "1387826781982400521-CiPnCO1nMb1isFPNCww5L7P02nmEDN"
access_secret = "MmM3ryNophADJgWM5PbwvpPOyTJH8hyi0wybljn4xIpK5"
# Get fully-trained XGBoostClassifier model
with open('model.pickle', 'rb') as read_file:
    xgb_model = pickle.load(read_file)

# Set up connection to Twitter API
# auth = tweepy.OAuthHandler(
#     twitter_keys['consumer_key'], twitter_keys['consumer_secret'])
# auth.set_access_token(
#     twitter_keys['access_token_key'], twitter_keys['access_token_secret'])
auth = tweepy.OAuthHandler(
    consumer_key, consumer_secret)
auth.set_access_token(
    access_key, access_secret)

print(auth)

api = tweepy.API(auth)
print(api)

def get_user_features(screen_name):
    '''
    Input: a Twitter handle (screen_name)
    Returns: a list of account-level information used to make a prediction 
            whether the user is a bot or not
    '''
    user = api.get_user(screen_name=screen_name)
    print(user.created_at)

    account_age_days = (datetime.now(timezone.utc) - user.created_at).days
    verified = user.verified
    geo_enabled = user.geo_enabled
    default_profile = user.default_profile
    default_profile_image = user.default_profile_image
    favourites_count = user.favourites_count
    followers_count = user.followers_count
    friends_count = user.friends_count
    statuses_count = user.statuses_count
    average_tweets_per_day = np.round(statuses_count / account_age_days, 3)
    hour_created = int(user.created_at.strftime('%H'))
    network = np.round(np.log(1 + friends_count)
                           * np.log(1 + followers_count), 3)
    tweet_to_followers = np.round(
            np.log(1 + statuses_count) * np.log(1 + followers_count), 3)
    follower_acq_rate = np.round(
            np.log(1 + (followers_count / account_age_days)), 3)
    friends_acq_rate = np.round(
            np.log(1 + (friends_count / account_age_days)), 3)
    account_features = [verified, hour_created, geo_enabled, default_profile, default_profile_image,
                            favourites_count, followers_count, friends_count, statuses_count,
                            average_tweets_per_day, network, tweet_to_followers, follower_acq_rate,
                            friends_acq_rate]
    # try:
    #     # Get user information from screen name
    #     user = api.get_user(screen_name)
    #     print(user)

        


    #     # # account features to return for predicton
    #     account_age_days = (datetime.now() - user.created_at).days
    #     verified = user.verified
    #     geo_enabled = user.geo_enabled
    #     default_profile = user.default_profile
    #     default_profile_image = user.default_profile_image
    #     favourites_count = user.favourites_count
    #     followers_count = user.followers_count
    #     friends_count = user.friends_count
    #     statuses_count = user.statuses_count
    #     average_tweets_per_day = np.round(statuses_count / account_age_days, 3)

    #     # # manufactured features
    #     hour_created = int(user.created_at.strftime('%H'))
    #     network = np.round(np.log(1 + friends_count)
    #                        * np.log(1 + followers_count), 3)
    #     tweet_to_followers = np.round(
    #         np.log(1 + statuses_count) * np.log(1 + followers_count), 3)
    #     follower_acq_rate = np.round(
    #         np.log(1 + (followers_count / account_age_days)), 3)
    #     friends_acq_rate = np.round(
    #         np.log(1 + (friends_count / account_age_days)), 3)

    #     # # organizing list to be returned
    #     account_features = [verified, hour_created, geo_enabled, default_profile, default_profile_image,
    #                         favourites_count, followers_count, friends_count, statuses_count,
    #                         average_tweets_per_day, network, tweet_to_followers, follower_acq_rate,
    #                         friends_acq_rate]

    # except:
    #     return 'User not found'

    return account_features if len(account_features) == 14 else f'User not found'
    # return f'User not found'


def bot_or_not(twitter_handle):
    '''
    Takes in a twitter handle and predicts whether or not the user is a bot
    Required: trained classification model (XGBoost) and user account-level info as features
    '''

    user_features = get_user_features(twitter_handle)

    if user_features == 'User not found':
        return 'User not found'

    else:
        # features for model
        features = ['verified', 'hour_created', 'geo_enabled', 'default_profile', 'default_profile_image',
                    'favourites_count', 'followers_count', 'friends_count', 'statuses_count', 'average_tweets_per_day',
                    'network', 'tweet_to_followers', 'follower_acq_rate', 'friends_acq_rate']

        # creates df for model.predict() format
        user_df = pd.DataFrame(np.matrix(user_features), columns=features)

        prediction = xgb_model.predict(user_df)[0]

        return "Bot" if prediction == 1 else "Not a bot"


def bot_proba(twitter_handle):
    '''
    Takes in a twitter handle and provides probabily of whether or not the user is a bot
    Required: trained classification model (XGBoost) and user account-level info from get_user_features
    '''
    user_features = get_user_features(twitter_handle)

    if user_features == 'User not found':
        return 'User not found'
    else:
        user = np.matrix(user_features)
        proba = np.round(xgb_model.predict_proba(user)[:, 1][0]*100, 2)
        return proba
