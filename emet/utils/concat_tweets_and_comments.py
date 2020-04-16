import pandas as pd
import numpy as np
import argparse

def command_line_parsing():
    """Parse command lines

    Parameters
    ----------
    train_path : str
        path to the train dataset
    output_path : str
        path to store the final file
    Returns
    -------
    parser
        The arguments from command line
    """
    parser = argparse.ArgumentParser(description = __doc__)

    parser.add_argument('--train-path', '-t',
                        dest='train_path',
                        required=True,
                        help='File path train dataset.')

    parser.add_argument('--output-path', '-o',
                        dest='output_path',
                        required=True,
                        help='File output path destination')

    return parser.parse_args()

def main():

    args = command_line_parsing()

    dataset = pd.read_pickle(args.train_path)

    print('dataset shape: ', dataset.shape)

    dataset.comments = [' '.join(c['comments']) for i, c in dataset.iterrows()]

    data_unique = dataset.drop_duplicates(subset='tweetId')
    index = data_unique.index
    print('unique tweets ids shape: ', data_unique.shape)
    rest_data = dataset.loc[~dataset.index.isin(index)]
    print('rest data shape: ', rest_data.shape)
    text = []
    comments = []

    for i in range(len(rest_data)):

        opts = data_unique[(data_unique.tweetId != rest_data.iloc[i, 7]) &
                         (data_unique.key_word == rest_data.iloc[i, 5]) &
                         (data_unique.label == rest_data.iloc[i, 6])].sample()

        tweetText = str(rest_data.iloc[i, 8] + ' ' + opts.tweetText.iloc[0])
        comment = str(rest_data.iloc[i, 10] + ' ' + opts.comments.iloc[0])

        text.append(tweetText)
        comments.append(comment)

    rest_data['tweetText'] = text
    rest_data['comments'] = comments

    print('final shape: ', rest_data.shape)

    rest_data.to_pickle(args.output_path)

if __name__ == '__main__':

    main()
