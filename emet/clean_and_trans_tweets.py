# --- PACKAGE IMPORTATION ---
from regular_expressions.pre_processing_texts import *
from translate.translate_post import *

import pandas as pd

# --- READ DATASET ---
dataset = pd.read_pickle('../new/embeded_tweets_and_bbcNews.pkl')

# --- REMOVE NOISE ---
def __preprocessing(tweet):

    p = Patterns()

    proc_tweet = p.remove_emoji_pattern(document=tweet)
    proc_tweet = p.hash_to_tag(proc_tweet, keep_words=True)
    proc_tweet = p.at_to_tag(proc_tweet, icon='')
    proc_tweet = p.url_to_tag(proc_tweet, icon='')
    proc_tweet = p.smileys_to_tag(proc_tweet, icon='')

    return proc_tweet

tweets = dataset.tweetText.apply(lambda tweet: __preprocessing(tweet))

# --- TRANSLATE ---
translator = TranslateTexts()
translator.translate_posts(posts=tweets, dest='en')

data = pd.DataFrame({'translated_tweet':translator.translated_posts})
data.to_csv('../new/translated_tweets.csv', index=False)
