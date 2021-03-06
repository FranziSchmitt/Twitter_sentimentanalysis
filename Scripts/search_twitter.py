#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 27 11:35:12 2018

@author: franzi
"""

#import tweepy
from tweepy import OAuthHandler, TweepError, API
from tweepy.error import RateLimitError
import json
import datetime as dt
import time
import os
import sys


def load_api():
    ''' Function that loads the twitter API after authorizing the user. '''
    
    tokens_dict = {}

    dir_base = os.path.dirname(os.path.realpath(__file__))
    dir_auth = os.path.join(dir_base, '../Twitter_authentication')
    file_auth = os.path.join(dir_auth, 'Authentication.txt')

    with open(file_auth, 'r') as secrets:
        for line in secrets:
            (key, value) = line.split()
            tokens_dict[key] = value
        

    auth = OAuthHandler(tokens_dict.get('consumer_key'), tokens_dict.get('consumer_secret'))
    auth.set_access_token(tokens_dict.get('access_token'), tokens_dict.get('access_secret'))
    
    # load the twitter API via tweepy
    return API(auth)

    
def tweet_search(api, query, max_tweets, max_id, since_id, lang='de'):
    ''' Function that takes in a search string 'query', the maximum
        number of tweets 'max_tweets', and the minimum (i.e., starting)
        tweet id. It returns a list of tweepy.models.Status objects. '''

    searched_tweets = []
    counter = 0
    while len(searched_tweets) < max_tweets:
        remaining_tweets = max_tweets - len(searched_tweets)
        counter += 1
        try:
            new_tweets = api.search(q=query, count=remaining_tweets,
                                    since_id=str(since_id),
				                    max_id=str(max_id-1),
                                   tweet_mode='extended', lang=lang)
#                                    geocode=geocode)
            
            print('found',len(new_tweets),'tweets {}'.format(query))
            if not new_tweets:
                print('no tweets found')
                break
            searched_tweets.extend(new_tweets)
            max_id = new_tweets[-1].id
        except TweepError:
            print('exception raised, waiting 15 minutes')
            print('(until:', dt.datetime.now()+dt.timedelta(minutes=15), ')')
            time.sleep(15*60)
            break # stop the loop
    return searched_tweets, max_id, counter


def get_tweet_id(api, date='', days_ago=9, query='a'):
    ''' Function that gets the ID of a tweet. This ID can then be
        used as a 'starting point' from which to search. The query is
        required and has been set to a commonly used word by default.
        The variable 'days_ago' has been initialized to the maximum
        amount we are able to search back in time (9).'''

    if date:
        # return an ID from the start of the given day
        td = date + dt.timedelta(days=1)
        tweet_date = '{0}-{1:0>2}-{2:0>2}'.format(td.year, td.month, td.day)
        tweet = api.search(q=query, count=1, until=tweet_date)
    else:
        # return an ID from __ days ago
        td = dt.datetime.now() - dt.timedelta(days=days_ago)
        tweet_date = '{0}-{1:0>2}-{2:0>2}'.format(td.year, td.month, td.day)
        # get list of up to 10 tweets
        try:
            tweet = api.search(q=query, count=10, until=tweet_date)
        except RateLimitError:
            print('Rate Limit Error raised')
            time.sleep(15*60)
            tweet = api.search(q=query, count=10, until=tweet_date)
        print('search limit (start/stop):',tweet[0].created_at)
        
        # return the id of the first tweet in the list
        return tweet[0].id


def write_tweets(tweets, filename):
    ''' Function that appends tweets to a file. '''

    with open(filename, 'a') as f:
        for tweet in tweets:
            json.dump(tweet._json, f)
            f.write('\n')


def main():
    ''' This is a script that continuously searches for tweets
        that were created over a given number of days. The search
        dates and search phrase can be changed below. '''

    ''' search variables: '''
    search_phrases = ['CSU', 'CDU', 
                     'SPD', 'Ltw_BY',
                     'AfD', 'ltw18',
                     'LTWBY', 'LTWBY18',
                     'Gruene','LINKE', 
                     'FDP', 'Grüne',
                     #'bayern', 'baden-wuerttenberg',
                     #'hessen', 'nrw', 'nordrhein-westfalen',
                     #'sachsen', 'sachsen-anhalt',
                     #'niedersachsen', 'mecklemburg-vorpommern',
                     #'thueringen', 'brandenburg', 'berlin', 'hamburg', 
                     #'bremen', 'rheinland-pfalz', 'saarland'
                    ]
    
    time_limit = 5                           # runtime limit in hours
    max_tweets = 100                           # number of tweets per search (will be
                                               # iterated over) - maximum is 100
    min_days_old, max_days_old = 1, 9         # search limits e.g., from 7 to 8
                                               # gives current weekday from last week,                                               # min_days_old=0 will search from right now
    #USA = '39.8,-95.583068847656,2500km'  
    #Germany = '51.163375, 10.447683, 500km'     # Germany includes all of Germany plus 
    #Bavaria = '48.7775, 11.431111, 200km'       # Bavaria includes all Bavaria plus
    

    # loop over search items,
    # creating a new file for each
    last_day = 2
    
    for max_days_old in range(last_day, 1, -1):
        min_days_old = max_days_old - 1 
            
        for search_phrase in search_phrases:

            print('Search phrase =', search_phrase)

            ''' other variables '''
            name = search_phrase.split()[0]
            dir_base = os.path.dirname(os.path.realpath(__file__))
            dir_save = os.path.join(dir_base, '../Data')
            json_file_root = os.path.join(dir_save,  name)
            
            os.makedirs(json_file_root, exist_ok=True)
            read_IDs = False
            
            # open a file in which to store the tweets
            # if scraped time span is exactly one day
            if max_days_old - min_days_old == 1:
                d = dt.datetime.now() - dt.timedelta(days=min_days_old)
                day = '{0}-{1:0>2}-{2:0>2}'.format(d.year, d.month, d.day)
            else:
                d1 = dt.datetime.now() - dt.timedelta(days=max_days_old-1)
                d2 = dt.datetime.now() - dt.timedelta(days=min_days_old)
                day = '{0}-{1:0>2}-{2:0>2}_to_{3}-{4:0>2}-{5:0>2}'.format(
                    d1.year, d1.month, d1.day, d2.year, d2.month, d2.day)
            file_name = name + '_' + day + '.json'
            json_file = os.path.join(json_file_root, file_name)
            # check if file for search term already exists
            if os.path.isfile(json_file):
                print('Appending tweets to file named: ',json_file)
                # flag for continue
                read_IDs = True
            
            # authorize and load the twitter API
            api = load_api()
            
            # set the 'starting point' ID for tweet collection
            if read_IDs:
                # open the json file and get the latest tweet ID
                with open(json_file, 'r') as f:
                    lines = f.readlines()
                    max_id = json.loads(lines[-1])['id']
                    print('Searching from the bottom ID in file')
            else:
                # get the ID of a tweet that is min_days_old
                if min_days_old == 0:
                    max_id = -1
                else:
                    max_id = get_tweet_id(api, days_ago=(min_days_old-1))
            # set the smallest ID to search for
            since_id = get_tweet_id(api, days_ago=(max_days_old-1))
            print('max id (starting point) =', max_id)
            print('since id (ending point) =', since_id)

            ''' tweet gathering loop  '''
            start = dt.datetime.now()
            end = start + dt.timedelta(hours=time_limit)
            count, exitcount = 0, 0
            while dt.datetime.now() < end:
                # collect tweets and update max_id
                tweets, max_id, internal_counter = tweet_search(api, search_phrase, max_tweets,
                                                                max_id=max_id, since_id=since_id
                                                                )
                count += internal_counter
                print('count =',count)
                # write tweets to file in JSON format
                if tweets:
                    write_tweets(tweets, json_file)
                    exitcount = 0
                else:
                    exitcount += 1
                    if exitcount == 3:
                        #if search_phrase == search_phrases[-1]:
                            #sys.exit('Maximum number of empty tweet strings reached - exiting')
                        #else:
                        print('Maximum number of empty tweet strings reached - breaking')
                        break
            
if __name__ == "__main__":
    main()
