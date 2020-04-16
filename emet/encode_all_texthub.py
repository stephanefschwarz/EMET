from generate_feature_vectors import multlingual_encoder
import pandas as pd
import sys
import csv
csv.field_size_limit(sys.maxsize)
import argparse

def command_line_parsing():
	"""Parse command lines

		Parameters
		----------
		train_path : str
			path to the train dataset
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
						help='Output path to store encoded embeddings.')

	return parser.parse_args()

def main():

	args = command_line_parsing()

	encoder = multlingual_encoder.MultilingualSentenceEncoder()

	data = pd.read_pickle(args.train_path)

	premises = data['bbc_news']
	hypothesis = data['tweetText']
	comments = [' '.join(comment['comments']) for index, comment in data.iterrows()]
	# comments = data['comments']

	print('Encoding premises...')
	premises = encoder.get_multilingual_embeddings(premises)
	print('Encoding hypothesis...')
	hypothesis = encoder.get_multilingual_embeddings(hypothesis)
	print('Encoding comments...')
	comments = encoder.get_multilingual_embeddings(comments)

	data['embedded_news'] = premises
	data['embedded_tweets'] = hypothesis
	data['embedded_comments'] = comments


	data.to_pickle(args.output_path)

if __name__ == '__main__':
    main()
