from hypothesis.get_tweets_comments import *
import pandas as pd
import numpy as np
import argparse
import sys
sys.setrecursionlimit(100000)

def command_line_parsing():

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--file-path', '-f',
                         dest='data_path',
                         required=True,
                         help='The path to the tweets IDs file')

    parser.add_argument('--output-file-path', '-o',
                         dest='output_path',
                         required=True,
                         help='The path and to the output pandas-pickle file')


    return parser.parse_args()

def scraping_twitter(args):
    dataset = pd.read_pickle(args.data_path)
    dataset['comments'] = None
    unique_ids = np.unique(dataset.tweetId)

    dataset['comments'] = dataset.tweetId.apply(lambda tw: get_comments(tw))

    dataset.to_pickle(args.output_path)

def bs4tag_to_text(args):

    tweet_comment = []
    comments = []
    d = pd.read_pickle(args.data_path)

    for com_list in d.comments:
        if (com_list == ['']):
            tweet_comment.append([''])
        else:
            for com in com_list:
                comments.append(com.text)

            tweet_comment.append(comments)
            comments = []

    d['comments'] = tweet_comment
    d.to_pickle(args.output_path)


    print(len(tweet_comment))
    print('\n===========================\n')
    print(tweet_comment[0:20])



if __name__ == '__main__':

    args = command_line_parsing()
    # scraping_twitter(args)
    bs4tag_to_text(args)
