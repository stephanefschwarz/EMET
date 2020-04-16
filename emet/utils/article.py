# ==== PACKAGE IMPORTATON ==== #
import pandas as pd
import numpy as np
import argparse
from sklearn import preprocessing
from sklearn.metrics import accuracy_score, f1_score
from emet.generate_feature_vectors import multlingual_encoder
from emet.models.cnn_arch import TrueCNN

def command_line_parsing():
	"""Parse command lines

		Parameters
		----------
		train_path : str
			path to the train dataset
		validation_path : str
			path to the validation dataset
		output_path : str
			path to store the new dataset
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

	parser.add_argument('--validation_path', '-v',
						dest='val_path',
						required=True,
						help='File path to the validation dataset.')

	parser.add_argument('--output_path', '-o',
						dest='output_path',
						required=True,
						help='Final destination of the new dataset.')

	return parser.parse_args()

def read_train_test(train_path, test_path):
	"""Read datasets pickle format

	Parameters
	----------
	train_path : str
		path to the train dataset
	test_path : str
		path to the validation dataset
	Returns
	-------
	pandas.DataFrame
		pandas.DataFrame for the two datasets
	"""
	return (pd.read_pickle(train_path), pd.read_pickle(test_path))

def join_comments(X_dataset):

	return [' '.join(comment['comments']) for index, comment in X_dataset.iterrows()]

def concatTEXT_train_dataset(X_train, n_samples=2,
                         consider_event=True,
                         consider_comments=True,
						 random=True,
                         output_path='./concatenated_TEXT_trainset.pkl'):
	"""Concatenate tweet TEXT of the training dataset based on the
	passed arguments.

	Parameters
	----------
	X_train : pandas.DataFrame
		an instace of the train dataset
	n_samples : int
		the wanted number of tweets to be
		concatenated, by default one two are merged.
	consider_event : boolean
		if the train set will be concatenated considering
		the post event
	consider_comments : boolean
		if the comments will be concatenated as well as
		the tweets text
	random : boolean
		if True the concatenation will be randonly, otherwise,
		will consider the tweet ID
	output_path : str
		final destination of the new train dataset
	Returns
	-------
	pandas.DataFrame
		concatenated samples
	"""
	def __get_random_sample(X_train, tweet, consider_event, consider_comments, random, n_samples):
		"""Method used to facilitate pandas apply function.
		Parameters
		----------
			Same of the outer Method
		Returns
		-------
			the new tweetText and comments of each train sample
		"""
		if (consider_event & random):

			samples = X_train[(X_train.key_word == tweet['key_word']) &
			(X_train.label == tweet['label'])].sample(n=n_samples,replace=True, random_state=1)

		elif (~consider_event & random):

			samples = X_train[X_train.label == tweet['label']].sample(n=n_samples, replace=True, random_state=1)

		elif (consider_event & ~random):

			samples = X_train[(X_train.key_word == tweet['key_word']) &
			(X_train.label == tweet['label']) &
			(X_train.tweetId != tweet['tweetId'])].sample(n=n_samples,replace=True, random_state=1)

		else: # (~consider_event & ~random)

			samples = X_train[(X_train.label == tweet['label']) &
			(X_train.tweetId != tweet['tweetId'])].sample(n=n_samples, replace=True, random_state=1)

		text = tweet['tweetText'] + ' ' + ' '.join(samples.tweetText)

		if (consider_comments):

			comment = tweet['comments'] + ' ' + ' '.join(samples.comments)

		else:
			comment = tweet['comments']

		return pd.Series((text, comment))

	# ============================================================================ #

	# if a single tweet has more than one comment
	# we need the concatenate them.
	X_train['comments'] = [' '.join(comment['comments']) for index, comment in X_train.iterrows()]

	X_train[['tweetText', 'comments']] = X_train.apply(lambda tw: __get_random_sample(X_train, tw, consider_event, consider_comments, random, n_samples), axis=1)

	X_train.to_pickle(output_path)

	return X_train

# ============================================================================ #
# ============================================================================ #

def get_label(predicted_labels, n_samples):
	"""Find the final label of a set of tweets
	predicted_labels : list
		predicted labels
	n_samples : int
		number of samples for each tweet
	"""

	final_labels = []

	for i in range(0, len(predicted_labels), n_samples):

		predic_per_sample = np.argmax(predicted_labels[i:i+n_samples], axis=1)

		labels, counts = np.unique(predic_per_sample, return_counts=True)

		final_labels.append(labels[np.argmax(counts)])

	return final_labels
# =============================================================================
# =============================================================================

def concatEMBE_train_dataset(X_train, n_samples=2,
                         consider_event=True,
                         consider_comments=True,
						 random=True,
						 method='mean',
                         output_path='./concatenated_EMB_trainset.pkl'):

	"""Concatenate tweet EMBEDDINGS of the training dataset based on the
	passed arguments.

	Parameters
	----------
	X_train : pandas.DataFrame
		an instace of the train dataset
	n_samples : int
		the wanted number of tweets to be
		concatenated, by default one two are merged.
	consider_event : boolean
		if the train set will be concatenated considering
		the post event
	consider_comments : boolean
		if the comments will be concatenated as well as
		the tweets text
	random : boolean
		if True the concatenation will be randonly, otherwise,
		will consider the tweet ID
	output_path : str
		final destination of the new train dataset
	Returns
	-------
	pandas.DataFrame
		concatenated samples
	"""
	def __get_random_sample(X_train, tweet, consider_event, consider_comments, random, n_samples, method):
		"""Method used to facilitate pandas apply function.

		Parameters
		----------
			Same of the outer Method
		Returns
		-------
			the new tweetText and comments of each train sample
		"""

		if (method == 'mean'):
			from numpy import mean as gen_feature
		elif (method == 'sum'):
			from numpy import sum as gen_feature
		else:
			from numpy import max as gen_feature

		if (consider_event & random):

			samples = X_train[(X_train.key_word == tweet['key_word']) &
			(X_train.label == tweet['label'])].sample(n=n_samples,replace=True, random_state=1)

		elif (~consider_event & random):

			samples = X_train[X_train.label == tweet['label']].sample(n=n_samples, replace=True, random_state=1)

		elif (consider_event & ~random):

			samples = X_train[(X_train.key_word == tweet['key_word']) &
			(X_train.label == tweet['label']) &
			(X_train.tweetId != tweet['tweetId'])].sample(n=n_samples,replace=True, random_state=1)

		else: # (~consider_event & ~random)

			samples = X_train[(X_train.label == tweet['label']) &
			(X_train.tweetId != tweet['tweetId'])].sample(n=n_samples, replace=True, random_state=1)

		emb_tweet = np.concatenate([tweet['embedded_tweets'], np.vstack(samples.embedded_tweets)])
		emg_tweet = gen_feature(emb_tweet, axis=0)

		if (consider_comments):

			emb_comments = np.concatenate([tweet['embedded_comments'], np.vstack(samples.embedded_comments)])
			emb_comments = gen_feature(emb_comments, axis=0)

		else:
			# emb_comments = gen_feature(emb_comments, axis=0)
			emb_comments = emb_comments

		return pd.Series((emg_tweet, emb_comments))

	# ====================================================================== #

	X_train[['embedded_tweets', 'embedded_comments']] = X_train.apply(lambda tw: __get_random_sample(X_train, tw, consider_event, consider_comments, random, n_samples, method), axis=1)
	X_train.to_pickle(output_path)

	return X_train

# ============================================================================ #
# ============================================================================ #

class TestSetAugmentation:
	"""
	Class to augmentate the test set, also used to transform the final labels.

	Attributes:
	----------
	X_train : pandas.DataFrame
		pandas dataframe of the training set
	X_test : pandas.DataFrame
		pandas dataframe of te test set
	samples : pandas.Dataframe
		real samples of the train set to be concatenated on the teste set
	samples_txt_real : pandas.DataFrame
		real samples to be concat. with fake and humor samples on test set
	samples_txt_fake : pandas.DataFrame
		fake samples to be concat. with real and humor samples on test set
	samples_txt_humor : pandas.DataFrame
		humor samples to be concat. with fake and real samples on test set
	Methods
	-------
	get_augumented_TEXT_testset(self, tweet, n_samples=5, consider_only_true=False, individual_aug=True, consider_comments=True)
		get an augmented testset from TEXT

	get_other_variables(tweet, n_samples)
		inner method to facilitate variables manipulation

	get_label(predicted_labels, n_samples)
		set the final label
	"""

	def __init__(self, X_train, X_test):

		self.X_train = X_train
		self.X_test = X_test

		self.samples_txt = None
		self.samples_txt_real = None
		self.samples_txt_fake = None
		self.samples_txt_humor = None

		self.sample_embedding = None
		self.sample_emb_real = None
		self.sample_emb_fake = None
		self.sample_emb_humor = None

		self.general_emb_real = None
		self.general_emb_fake = None
		self.general_emb_humor = None

	def get_other_variables(self, tweet, n_samples, text=False):

		"""Used to generate the other variable that will be not changed

		Parameters
		----------
		tweet : pandas.DataFrame
			a single sample of testset
		n_samples : int
			number of samples
		"""
#
		bbc_news = [tweet['bbc_news']] * n_samples
		imageId = [tweet['imageId']] * n_samples
		key_word = [tweet['key_word']] * n_samples
		label = [tweet['label']] * n_samples
		tweetId = [tweet['tweetId']] * n_samples
		url = [tweet['url']] * n_samples

		if (text):
			tweetText = [tweet['tweetText']] * n_samples

			return pd.DataFrame({'tweetId':tweetId,
						 'tweetText': tweetText,
						 'label':label,
						 'bbc_news':bbc_news,
						 'imageId':imageId,
						 'key_word':key_word,
						 'url':url})

		return pd.DataFrame({'tweetId':tweetId,
							 'label':label,
							 'bbc_news':bbc_news,
							 'imageId':imageId,
							 'key_word':key_word,
							 'url':url})

	# =================================================================
	# =================================================================

	def get_augumented_TEXT_testset(self, tweet, n_samples=5, consider_only_true=False, individual_aug=True, consider_comments=True):
		"""augmente the test set

		Parameters
		----------
		tweet : pandas.Series
			a single sample of the test set
		n_samples : int
			number of samples to get from trainset
		consider_only_true : boolean
			if needs to consider only true news
		individual_aug : boolean
			if is to generate individual random samplings for each testset sample OR one set of sample for all testset
		consider_comments : boolean
			concatenate or not the comments.

		Returns
		-------
		pandas.DataFrame
			a new testset
		"""

		def __get_comments(self, tweet, samples, consider_comments):
			"""get the new list of comments to use on the new teste set

			Parameters
			----------
			tweet : pandas.DataFrame
			a single tweet from the testset
			consider_comments : boolean
			concatenate of not all the comments

			Returns
			-------
			list
			a list of comments
			"""

			new_comments = []

			if (consider_comments):

				new_comments.extends([tweet.comments + ' ' + ' '.join(comment) for comment in samples['comments']])
			else:
				new_comments = [tweet['comments']] * len(samples)

			return new_comments

		# =================================================================
		# =================================================================

		if (consider_only_true & ~individual_aug):

			if (self.samples_txt == None):

				self.samples_txt = self.X_train[self.X_train.label == 'real'].samples(n=n_samples, replace=True, random_state=1)

			new_dataset = self.get_other_variables(tweet, n_samples)

			tweet_text.extend([tweet['tweetText'] + ' ' + ' '.join(sample_txt for sample_txt in self.samples_txt)])

			new_dataset['tweetText'] = tweetText

			new_dataset['comments'] = __get_comments(tweet_text, self.samples_txt, consider_comments)

			return new_dataset

		if (~consider_only_true & ~individual_aug):

			if (self.samples_txt_real == None):

				self.samples_true = self.X_train[self.X_train.label == 'real'].samples(n=n_samples, replace=True, random_state=1)

				self.samples_txt_fake = self.X_train[self.X_train.label == 'fake'].samples(n=n_samples, replace=True, random_state=1)

				self.samples_txt_humor = self.X_train[self.X_train.label == 'humor'].samples(n=n_samples, replace=True, random_state=1)

			tweet_text.extend([tweet['tweetText'] + ' ' + ' '.join(sample_txt for sample_txt in self.samples_txt_real)])

			tweet_text.extend([tweet['tweetText'] + ' ' + ' '.join(sample_txt for sample_txt in self.samples_txt_fake)])

			tweet_text.extend([tweet['tweetText'] + ' ' + ' '.join(sample_txt for sample_txt in self.samples_txt_humor)])

			new_dataset = self.get_other_variables(tweet, n_samples*3)

			new_dataset['tweetText'] = tweet_text
			new_dataset['comments'] = __get_comments(tweet_text, pd.concat([self.samples_true, self.samples_txt_fake,self.samples_txt_humor]), consider_comments)

			return new_dataset

		if (consider_only_true & individual_aug):

			samples = self.X_train.tweetText[self.X_train.label == 'real'].samples(n=n_samples, replace=True, random_state=1)

			tweet_text.extend([tweet['tweetText'] + ' ' + ' '.join(sample_txt for sample_txt in self.samples_txt)])

			new_dataset = self.get_other_variables(tweet, n_samples)
			new_dataset['comments'] = __get_comments(tweet_text, samples, consider_comments)
			new_dataset['tweetText'] = tweet_text

			return new_dataset

		if (~consider_only_true & individual_aug):

			samples_true = self.X_train[self.X_train.label == 'real'].samples(n=n_samples, replace=True, random_state=1)

			samples_txt_fake = self.X_train[self.X_train.label == 'fake'].samples(n=n_samples, replace=True, random_state=1)

			samples_txt_humor = self.X_train[self.X_train.label == 'humor'].samples(n=n_samples, replace=True, random_state=1)

			tweet_text.extend([tweet['tweetText'] + ' ' + ' '.join(sample_txt for sample_txt in samples_txt_real)])

			tweet_text.extend([tweet['tweetText'] + ' ' + ' '.join(sample_txt for sample_txt in samples_txt_fake)])

			tweet_text.extend([tweet['tweetText'] + ' ' + ' '.join(sample_txt for sample_txt in samples_txt_humor)])

			new_dataset = self.get_other_variables(tweet, n_samples)
			new_dataset['comments'] = __get_comments(tweet_text, pd.concat([samples_true,samples_txt_fake,samples_txt_humor]), consider_comments)
			new_dataset['tweetText'] = tweet_text

			return new_dataset

# =============================================================================
# =============================================================================

	def get_augumented_EMB_testset(self, tweet, n_samples=None, consider_only_true=False, individual_aug=True, consider_comments=True, method='max'):
		"""Generate augumented dataset from text embeddings

		Parameters
		----------
		tweet : pandas.Series
			a single sample of the test set
		n_samples : int
			number of samples to get from trainset, if None, get the mean based on all train set
		consider_only_true : boolean
			if needs to consider only true news
		individual_aug : boolean
			if is to generate individual random samplings for each testset sample OR one set of sample for all testset
		consider_comments : boolean
			concatenate or not the comments.
		method : str
			method used to merge the tweets. Acceptable methods: mean, min, max and sum.

		Returns
		-------
		pandas.DataFrame
			a new testset
		"""
		# =====================================================================
		# =====================================================================

		def __get_comments(self, tweet, samples, n_samples, consider_comments=True):

			print(type(samples))
			print(samples)
			exit()

			tweet_emb = np.vstack(tweet.embedded_comments)
			samples_emb = np.vstack(samples.embedded_comments)
			comments = []

			if (consider_comments):

				comments.append(np.sum([tweet_emb, samples_emb], axis=0))
				comemnts = np.array(comments)
				comments = comments.reshape((comments.shape[0], comments.shape[2]))
				return comments

			else:
				return [[tweet_emb] * n_samples]

		# =====================================================================
		# =====================================================================

		aug_emb_tweets = []

		if (n_samples == None):

			if (self.general_emb_real is None):

				self.general_emb_real = np.mean(np.vstack(self.X_train.embedded_tweets[self.X_train.label == 'real']), axis=0)
				self.general_emb_real = self.general_emb_real.reshape((self.general_emb_real.shape[0], 1))

				self.general_emb_fake = np.mean(np.vstack(self.X_train.embedded_tweets[self.X_train.label == 'fake']), axis=0)
				self.general_emb_fake = self.general_emb_fake.reshape((self.general_emb_fake.shape[0], 1))

				self.general_emb_humor = np.mean(np.vstack(self.X_train.embedded_tweets[self.X_train.label == 'humor']), axis=0)
				self.general_emb_humor = self.general_emb_humor.reshape((self.general_emb_humor.shape[0], 1))

			aug_emb_tweets.append(np.sum([self.general_emb_real, np.vstack(tweet['embedded_tweets'])], axis=0))
			aug_emb_tweets.append(np.sum([self.general_emb_fake, np.vstack(tweet['embedded_tweets'])], axis=0))
			aug_emb_tweets.append(np.sum([self.general_emb_humor, np.vstack(tweet['embedded_tweets'])], axis=0))

			dataset = self.get_other_variables(tweet, 3, text=True)
			dataset['embedded_tweets'] = aug_emb_tweets

			comments_samples = pd.concat([self.X_train.embedded_comments[self.X_train.label == 'real'], self.X_train.embedded_comments[self.X_train.label == 'fake'], self.X_train.embedded_comments[self.X_train.label == 'humor']])
			dataset['embedded_comments'] = __get_comments(tweet=tweet, samples=comments_samples, n_samples=3, consider_comments=consider_comments)

			return dataset

		if (consider_only_true & ~individual_aug):

			if (self.sample_embedding is None):

				self.sample_embedding = self.X_train[self.X_train.label == 'real'].sample(n=n_samples, replace=True, random_state=1)

			sample_embedding = np.mean(np.stack(self.samples_embedding.embedded_tweets), axis=0)
			sample_embedding = sample_embedding.reshape((sample_embedding.shape[0], 1))

			dataset = self.get_other_variables(tweet, 1, text=True)
			dataset['embedded_tweets'] = np.sum([sample_embedding, np.vstack(tweet['embedded_tweets'])], axis=0)
			dataset['embedded_comments'] = __get_comments(tweet=tweet, samples=self.samples_embedding.embedded_comments, n_samples=1, consider_comments=consider_comments)

			return dataset

		if (~consider_only_true & ~individual_aug):

			if (self.sample_emb_real is None):

				self.sample_emb_real = self.X_train[(self.X_train.label == 'real')].sample(n=n_samples, replace=True, random_state=1)

				self.sample_emb_fake = self.X_train[(self.X_train.label == 'fake')].sample(n=n_samples, replace=True, random_state=1)

				self.sample_emb_humor = self.X_train[(self.X_train.label == 'humor')].sample(n=n_samples, replace=True, random_state=1)

			sample_emb_real = np.mean(np.stack(self.sample_emb_real.embedded_tweets), axis=0)
			sample_emb_real = sample_emb_real.reshape((sample_emb_real.shape[1], 1))

			sample_emb_fake = np.mean(np.stack(self.sample_emb_fake.embedded_tweets), axis=0)
			sample_emb_fake = sample_emb_fake.reshape((sample_emb_fake.shape[1], 1))

			sample_emb_humor = np.mean(np.stack(self.sample_emb_humor.embedded_tweets), axis=0)
			sample_emb_humor = sample_emb_humor.reshape((sample_emb_humor.shape[1], 1))

			aug_emb_tweets.append(np.sum([sample_emb_real, np.vstack(tweet['embedded_tweets'])], axis=0))
			aug_emb_tweets.append(np.sum([sample_emb_fake, np.vstack(tweet['embedded_tweets'])], axis=0))
			aug_emb_tweets.append(np.sum([sample_emb_humor, np.vstack(tweet['embedded_tweets'])], axis=0))

			dataset = self.get_other_variables(tweet, 3, text=True)
			dataset['embedded_tweets'] = aug_emb_tweets
			comments = pd.concat([self.sample_emb_real,self.sample_emb_fake,self.sample_emb_humor])
			print('\n\n COMMENTS: ', comments, '\n\n')
			dataset['embedded_comments'] = __get_comments(tweet, comments, 3, consider_comments)


			return dataset

		if (consider_only_true & individual_aug):

			sample = self.X_train[self.X_train.label == 'real'].sample(n=n_samples, replace=True, random_state=1)
			sample_embedding = np.mean(np.stack(sample.embedded_tweets))
			sample_embedding = sample_embedding.reshape((sample_embedding.shape[0], 1))
			aug_emb_tweets = np.sum([sample_embedding, np.vstack(tweet['embedded_tweets'])], axis=0)

			dataset = self.get_other_variables(tweet, 1, text=True)
			dataset['embedded_tweets'] = aug_emb_tweets
			dataset['embedded_comments'] = __get_comments(tweet, sample.embedded_comments, 1, consider_comments)

			return dataset


		if (~consider_only_true & individual_aug):

			sample_real =self.X_train[self.X_train.label == 'real'].sample(n=n_samples, replace=True, random_state=1)
			sample_fake = self.X_train[self.X_train.label == 'fake'].sample(n=n_samples, replace=True, random_state=1)
			sample_humor = self.X_train[self.X_train.label == 'humor'].sample(n=n_samples, replace=True, random_state=1)


			sample_emb_real = np.mean(np.stack(sample_real), axis=0)
			sample_emb_real = sample_emb_real.reshape((sample_emb_real.shape[0], 1))

			sample_emb_fake = np.mean(np.stack(sample_fake), axis=0)
			sample_emb_fake = sample_emb_fake.reshape((sample_emb_fake.shape[0], 1))

			sample_emb_humor = np.mean(np.stack(sample_humor), axis=0)
			sample_emb_humor = sample_emb_humor.reshape((sample_emb_humor.shape[0], 1))

			aug_emb_tweets.append(np.sum([sample_emb_real, np.vstack(tweet['embedded_tweets'])], axis=0))
			aug_emb_tweets.append(np.sum([sample_emb_fake, np.vstack(tweet['embedded_tweets'])], axis=0))
			aug_emb_tweets.append(np.sum([sample_emb_humor, np.vstack(tweet['embedded_tweets'])], axis=0))

			dataset = self.get_other_variables(tweet, 3, text=True)
			dataset['embedded_tweets'] = aug_emb_tweets
			comments = pd.concat([sample_real, sample_fake, sample_humor])
			dataset['embedded_comments'] = __get_comments(tweet, comments, 3, consider_comments)

			return dataset
