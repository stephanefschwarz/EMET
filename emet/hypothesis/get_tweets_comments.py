import requests as r
from bs4 import BeautifulSoup as bs
import sys
sys.setrecursionlimit(100000)

def get_comments(tweet_id):
    """Extract Twitter comments from a specific
       tweet using for that the tweet ID

    Parameters
    ----------
    tweet_id : int
        a single tweet ID

    Returns
    -------
    list
        a list of comments from the tweets or None if the
        tweets has no comments of didn't exists anymore
    """
    tweet_page = r.get('https://twitter.com/itz_rubby/status/'+str(tweet_id))
    page = bs(tweet_page.content, 'html.parser')

    comments = page.find_all("p", attrs={"class":"TweetTextSize js-tweet-text tweet-text"})

    if(len(comments) > 0):

        return comments
    else:
        return ['']
