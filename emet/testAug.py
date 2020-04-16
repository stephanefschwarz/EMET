import pandas as pd
import numpy as np


def command_line_parsing():

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--file-path', '-f',
                         dest='train_path',
                         required=True,
                         help='The path to the tweets IDs file')

    parser.add_argument('--test-path', '-t',
                         dest='test_path',
                         required=True,
                         help='The path to test dataset')

    parser.add_argument('--output-file-path', '-o',
                         dest='output_path',
                         required=True,
                         help='The path and to the output pandas-pickle file')


    return parser.parse_args()

def get_news_mean(X_train):

    real_news_mean = np.mean(X_train.embedded_news[X_train.label == 'real'])
    fake_news_mean = np.mean(X_train.embedded_news[X_train.label == 'fake'])
    humor_news_mean = np.mean(X_train.embedded_news[X_train.label == 'humor'])

    return real_news_mean, fake_news_mean, humor_news_mean


if __name__ == '__main__':

    args = command_line_parsing()

    X_train = pd.read_pickle(args.train_path)
    X_test = pd.read_pickle(args.test_path)

    real, fake, humor = get_news_mean(X_train)

    

    main()
