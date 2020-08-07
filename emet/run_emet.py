from generate_feature_vectors import multlingual_encoder
from utils import utils
from sklearn.metrics import accuracy_score, f1_score
from models.cnn_arch import TrueCNN
from sklearn import preprocessing
import pandas as pd
import numpy as np
import argparse
import sys
from keras.callbacks import Callback
import matplotlib.pyplot as plt
from os import system
from sklearn.metrics import precision_recall_fscore_support

def command_line_parsing():
	"""Parse command lines

		Parameters
		----------
		train_path : str
			path to the train dataset
		validation_path : str
			path to the validation dataset
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

	return parser.parse_args()

def get_new_test_samples(X_train, n_samples, sample):

	new_tweetText = []
	steps = n_samples * 3
	new_news = [sample['embedded_news']] * steps
	new_comments = [sample['embedded_comments']] * steps
	new_labels = [sample['label']] * steps

	random_real_samples = X_train['tweetText'][X_train.label == 'real'].sample(n=n_samples,
																		  random_state=1)
	random_fake_samples = X_train['tweetText'][X_train.label == 'fake'].sample(n=n_samples,
																		  random_state=1)
	random_humor_samples = X_train['tweetText'][X_train.label == 'humor'].sample(n=n_samples,
																		  random_state=1)

	new_tweetText.extend([str(sample['tweetText'] + ' ' + s) for s in random_real_samples])
	new_tweetText.extend([str(sample['tweetText'] + ' ' + s) for s in random_fake_samples])
	new_tweetText.extend([str(sample['tweetText'] + ' ' + s) for s in random_humor_samples])


	# encoder = multlingual_encoder.MultilingualSentenceEncoder()
	# new_test_samples = encoder.get_multilingual_embeddings(new_tweetText)
	# new_test_samples = np.vstack(new_test_samples)
	# new_test_samples = np.reshape(new_test_samples.shape[0],
	# 							  new_test_samples.shape[1], 1)

	final_dataframe = pd.DataFrame({'tweetText' : new_tweetText,
									'embedded_news' : new_news,
									# 'embedded_tweets' : new_test_samples,
									'embedded_comments' : new_comments,
									'label' : new_labels})
	return final_dataframe

def get_mean(X_test, n_samples):
	news = []
	tweets = []
	comments = []
	labels = []

	for i in range(0, X_test.shape[0], n_samples):

		mean = np.mean(np.vstack(X_test.embedded_tweets[i:i+n_samples]), axis=0)
		news.append(X_test.embedded_news.iloc[i])
		tweets.append(mean)
		comments.append(X_test.embedded_comments.iloc[i])
		labels.append(X_test.label.iloc[i])

	return pd.DataFrame({'embedded_news':news,
						 'embedded_tweets':tweets,
						 'embedded_comments':comments,
						 'label':labels})

# def classify(train_path, X_test_path):
#
# 	sample = 1
# 	new_X_test = pd.DataFrame()
# 	X_train = pd.read_pickle(train_path)
# 	X_test = pd.read_pickle(X_test_path)
#
# 	test_samples = X_test.apply(lambda tw: get_new_test_samples(X_train, sample, tw, axis=1)
# 	# new_X_test = pd.concat([d for d in samples])
# 	# new_X_test.to_pickle('./dataset/aug_testset_1samples.pkl')
# 	# encoder = multlingual_encoder.MultilingualSentenceEncoder()
# 	# new_X_test['embedded_tweets'] = encoder.get_multilingual_embeddings(new_X_test['tweetText'])
# 	# new_X_test.to_pickle('./dataset/aug_testset_1samples.pkl')
# 	# new_X_test = pd.read_pickle('./dataset/aug_testset_10samples.pkl')
# 	# new_X_test = get_mean(new_X_test, 10)
# 	# new_X_test = pd.read_pickle('./dataset/mean_testset_5samples.pkl')
#
# 	X_news_test = np.vstack(new_X_test.embedded_news)
# 	X_tweet_test = np.vstack(new_X_test.embedded_tweets)
# 	X_comments_test = np.vstack(new_X_test.embedded_comments)
# 	X_news_test = X_news_test.reshape((X_news_test.shape[0], X_news_test.shape[1], 1))
# 	X_tweet_test = X_tweet_test.reshape((X_tweet_test.shape[0], X_tweet_test.shape[1], 1))
# 	X_comments_test = X_comments_test.reshape((X_comments_test.shape[0], X_comments_test.shape[1], 1))
#
# 	X_news_train = np.vstack(X_train.embedded_news)
# 	X_tweet_train = np.vstack(X_train.embedded_tweets)
# 	X_comments_train = np.vstack(X_train.embedded_comments)
# 	X_news_train = X_news_train.reshape((X_news_train.shape[0], X_news_train.shape[1], 1))
# 	X_tweet_train = X_tweet_train.reshape((X_tweet_train.shape[0], X_tweet_train.shape[1], 1))
# 	X_comments_train = X_comments_train.reshape((X_comments_train.shape[0], X_comments_train.shape[1], 1))
#
# 	label_encoder = preprocessing.LabelBinarizer()
# 	label_encoder.fit_transform(X_train.label)
# 	y_train = label_encoder.transform(X_train.label)
# 	y_test = label_encoder.transform(X_test.label)
#
# 	tCNN = TrueCNN(news_input_shape=(X_news_train.shape[1], 1),
# 		           tweet_input_shape=(X_tweet_train.shape[1], 1),
# 		           comments_input_shape=(X_comments_train.shape[1], 1)).model
#
# 	tCNN.fit(x=[X_news_train, X_tweet_train, X_comments_train], y=y_train, batch_size=40, epochs=10)
#
# 	predictions = tCNN.predict([X_news_test, X_tweet_test, X_comments_test], batch_size=10)
#
# 	expected = np.argmax(y_test, axis=1)
# 	predicted = set_final_label(predictions, n_samples=sample*3)
#
# 	print('Accuracy: ', accuracy_score(predicted, expected, normalize=True),
# 		  'F1 score: ', f1_score(predicted, expected, average='weighted'))


def get_testset_mean(X_train, X_news_test, X_tweet_test, X_comments_test, test_labels):

	# from numpy import mean as fusion
	# from numpy import sum as fusion
	from numpy import max as fusion

	new_X_news_test = []
	new_X_tweet_test = []
	new_X_comments_test = []
	new_label = []

	# mean_new_real = np.mean(np.vstack(X_train.embedded_news[X_train.label == 'real']), axis=0)
	# mean_new_real = mean_new_real.reshape((mean_new_real.shape[0], 1))
	#
	# mean_new_fake = np.mean(np.vstack(X_train.embedded_news[X_train.label == 'fake']), axis=0)
	# mean_new_fake = mean_new_fake.reshape((mean_new_fake.shape[0], 1))
	#
	# mean_new_humor = np.mean(np.vstack(X_train.embedded_news[X_train.label == 'humor']), axis=0)
	# mean_new_humor = mean_new_humor.reshape((mean_new_humor.shape[0], 1))
	# ---------------------------------------------------- #
	# ---------------------------------------------------- #
	mean_tweet_real = fusion(np.vstack(X_train.embedded_tweets[X_train.label == 'real']), axis=0)
	mean_tweet_real = mean_tweet_real.reshape((mean_tweet_real.shape[0], 1))

	mean_tweet_fake = np.mean(np.vstack(X_train.embedded_tweets[X_train.label == 'fake']), axis=0)
	mean_tweet_fake = mean_tweet_fake.reshape((mean_tweet_fake.shape[0], 1))

	mean_tweet_humor = np.mean(np.vstack(X_train.embedded_tweets[X_train.label == 'humor']), axis=0)
	mean_tweet_humor = mean_tweet_humor.reshape((mean_tweet_humor.shape[0], 1))
	# ---------------------------------------------------- #
	# ---------------------------------------------------- #
	# mean_comments_real = np.mean(np.vstack(X_train.embedded_comments[X_train.label == 'real']), axis=0)
	# mean_comments_real = mean_comments_real.reshape((mean_comments_real.shape[0], 1))
	#
	# mean_comments_fake = np.mean(np.vstack(X_train.embedded_comments[X_train.label == 'fake']), axis=0)
	# mean_comments_fake = mean_comments_fake.reshape((mean_comments_fake.shape[0], 1))
	#
	# mean_comments_humor = np.mean(np.vstack(X_train.embedded_comments[X_train.label == 'humor']), axis=0)
	# mean_comments_humor = mean_comments_humor.reshape((mean_comments_humor.shape[0], 1))

	# ---------------------------------------------------- #

	for i in range(X_news_test.shape[0]):

		# new_X_news_test.append(np.sum([mean_new_real, X_news_test[i]], axis=0))
		# new_X_news_test.append(np.mean([mean_new_fake, X_news_test[i]], axis=0))
		# new_X_news_test.append(np.mean([mean_new_humor, X_news_test[i]], axis=0))

		# -------------------
		new_X_tweet_test.append(np.sum([mean_tweet_real, X_tweet_test[i]], axis=0))
		new_X_tweet_test.append(np.sum([mean_tweet_fake, X_tweet_test[i]], axis=0))
		new_X_tweet_test.append(np.sum([mean_tweet_humor, X_tweet_test[i]], axis=0))
		#
		new_X_news_test.append(X_news_test[i])
		new_X_news_test.append(X_news_test[i])
		new_X_news_test.append(X_news_test[i])
		# -------------------

		# new_X_comments_test.append(np.sum([mean_comments_real, X_comments_test[i]], axis=0))
		# new_X_comments_test.append(np.mean([mean_comments_fake, X_comments_test[i]], axis=0))
		# new_X_comments_test.append(np.mean([mean_comments_humor, X_comments_test[i]], axis=0))

		# -------------------
		new_X_comments_test.append(X_comments_test[i])
		new_X_comments_test.append(X_comments_test[i])
		new_X_comments_test.append(X_comments_test[i])
		# -------------------

		new_label.append(test_labels[i].tolist())
		new_label.append(test_labels[i].tolist())
		new_label.append(test_labels[i].tolist())

	return new_X_news_test, new_X_tweet_test, new_X_comments_test, new_label

def set_final_label(new_label, n_samples=3):

	labels = []
	for i in range(0, len(new_label), n_samples):

		# labels.append(np.argmax(new_label[i:i+2], axis=1)[0])
		options = np.argmax(new_label[i:i+n_samples], axis=1)
		label, counts = np.unique(options, return_counts=True)
		final_label = label[np.argmax(counts)]
		labels.append(final_label)

	return labels

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

def new_label_func(X_test, predicted, excected):

	X_test = X_test.reset_index(drop=True)

	unique_id = np.unique(X_test.tweetId)

	ground_truth = []
	pred = []

	for id in unique_id:
		index = X_test[X_test.tweetId == id].index.values[0]
		ground_truth.append(excected[index])
		values = X_test[X_test.tweetId == id].index.values
		pred_labels = np.array(predicted)[values]
		uni_labels, counts = np.unique(pred_labels, return_counts=True)
		pred.append(uni_labels[np.argmax(counts)])

	return ground_truth, pred


def main():

	system('clear')

	args = command_line_parsing()

	# classify(args.train_path, args.val_path)
	# exit()

	X_train = pd.read_pickle(args.train_path)
	X_test = pd.read_pickle(args.val_path)
	# 92.39 evaluate, acc: 90.16, F1: 90.51
	X_test = X_test[X_test.key_word.isin(['Solar Eclipse', 'Garissa Attack', 'Nepal earthquake', 'Samurai and Girl','syrian boy beach', 'Varoufakis and ZDF'])]
	#
	# X_test = X_test[X_test.key_word.isin(['airstrikes','american soldier quran',
	# 									  'ankara explosions', 'attacks paris',
	# 									  'black lion','boko haram','bowie david',
	# 									  'brussels car metro','brussels explosions',
	# 									  'burst kfc', 'bush book','convoy explosion turkey',
	# 									  'donald trump attacker', 'eagle kid',
	# 									  'five headed snake','fuji','gandhi dancing',
	# 									  'half everything','hubble telescope',
	# 									  'immigrants','isis children', 'john guevara',
	# 									  'McDonalds fee','nazi submarine','north korea',
	# 									  'not afraid','pakistan explosion','pope francis',
	# 									  'protest', 'refugees','rio moon','snowboard girl',
	# 									  'soldier stealing', 'syrian children',
	# 									  'ukrainian nazi','woman 14 children'])]



	# X_train = concatEMBE_train_dataset(X_train, n_samples=1,
	# 			                         consider_event=False,
	# 			                         consider_comments=True,
	# 									 random=False,
	# 									 method='sum',
	# 	                         output_path='MediaEval_feature_extraction/dataset/con1Tw_EveFalse_EMB_trainset.pkl'
	# 									 )

	X_news_train = np.vstack(X_train.embedded_news)
	print(X_news_train.shape)
	X_tweet_train = np.vstack(X_train.embedded_tweets)
	X_comments_train = np.vstack(X_train.embedded_comments)

	X_news_train = X_news_train.reshape((X_news_train.shape[0], X_news_train.shape[1], 1))
	print(X_news_train.shape)

	X_tweet_train = X_tweet_train.reshape((X_tweet_train.shape[0], X_tweet_train.shape[1], 1))
	X_comments_train = X_comments_train.reshape((X_comments_train.shape[0], X_comments_train.shape[1], 1))
	
	X_news_test = np.vstack(X_test.embedded_news)
	X_tweet_test = np.vstack(X_test.embedded_tweets)
	X_comments_test = np.vstack(X_test.embedded_comments)

	X_news_test = X_news_test.reshape((X_news_test.shape[0], X_news_test.shape[1], 1))
	X_tweet_test = X_tweet_test.reshape((X_tweet_test.shape[0], X_tweet_test.shape[1], 1))
	X_comments_test = X_comments_test.reshape((X_comments_test.shape[0], X_comments_test.shape[1], 1))

	# ---------------------------------------------------- #
	# USER THIS IN CASE OF THE FINAL DENSE LAYER WAS SHAPE OF 1
	#
	# label_encoder = preprocessing.LabelEncoder()
	# label_encoder.fit(X_train.label)
	# y_train = label_encoder.transform(X_train.label)
	# y_test = label_encoder.transform(X_test.label)
	# y_train = y_train.reshape((X_train.shape[0],1))
	# y_test = y_test.reshape((X_test.shape[0], 1))
	#
	# ---------------------------------------------------- #

	# ---------------------------------------------------- #
	# AS I NEED PROBABILITIES, I YOU CHAGE THE FINAL DENSE
	# LAYER TO BE SHAPE OF (#SAMPLES, 3), WHERE 3 REPRESENTS
	# ALL OF MY CLASSES.

	label_encoder = preprocessing.LabelBinarizer()
	label_encoder.fit_transform(X_train.label)

	y_train = label_encoder.transform(X_train.label)
	y_test = label_encoder.transform(X_test.label)
	# (new_X_news_test, new_X_tweet_test,
	#  new_X_comments_test, new_label) = get_testset_mean(X_train,
	#  													X_news_test,
	# 													X_tweet_test,
	# 													X_comments_test,
	# 													y_test)


	new_X_news_test = X_news_test
	new_X_tweet_test = X_tweet_test
	new_X_comments_test = X_comments_test
	new_label = y_test

	x_test = [new_X_news_test, new_X_tweet_test, new_X_comments_test]
	new_label = np.array(new_label)

	# ---------------------------------------------------- #
	class TestCallback(Callback):
		acc_history = []
		def __init__(self, test_data):
			self.test_data = test_data

		def on_epoch_end(self, epoch, logs=None):

			X_test, y_test = self.test_data

			X_news_test = x_test[0]
			X_tweet_test = x_test[1]
			X_comments_test = x_test[2]

			eval = self.model.evaluate([X_news_test, X_tweet_test
			, X_comments_test
			], y_test)

			self.acc_history.append(eval[1])


	tCNN = TrueCNN(news_input_shape=(X_news_train.shape[1], 1),
		           tweet_input_shape=(X_tweet_train.shape[1], 1)
				   # ,comments_input_shape=(X_comments_train.shape[1], 1)
				   ).model

	# history = tCNN.fit(x=[X_news_train, X_tweet_train, X_comments_train], y=y_train, batch_size=40, epochs=1)

	# print('Evaluate: ', tCNN.evaluate([new_X_news_test, new_X_tweet_test, new_X_comments_test], new_label))
	print('CALLBACK---------')
	call = TestCallback((x_test, new_label))
	print('history --- ')
	history = tCNN.fit(x=[X_news_train, X_tweet_train
	# , X_comments_train
	], y=y_train,
	# callbacks=[call],
	batch_size=40, epochs=10)
	print(tCNN.evaluate([X_news_test, X_tweet_test
	# , X_comments_test
	], y_test))

	expected = np.argmax(new_label, axis=1)
	print('predictions --')
	predictions = tCNN.predict([new_X_news_test, new_X_tweet_test
	# , new_X_comments_test
	])
	predicted = np.argmax(predictions, axis=1)

	print(precision_recall_fscore_support(expected, predicted, average='weighted'))

	# print('Eval : ',call.acc_history)
	# print('Train : ',history.history['accuracy'])


	# plt.plot(history.history['accuracy'], 'g-', label='Train')
	# plt.plot(call.acc_history, 'b-', label='Test')
	# plt.axhline(y=0.9, color='r', linestyle='--')
	#
	# plt.ylabel('accuracy')
	# plt.xlabel('epoch')
	# # plt.legend(['train', 'test'], loc='upper right')
	# plt.legend()
	# plt.savefig('MediaEval_feature_extraction/dataset/checked_comments.png')

	exit()

	history = tCNN.fit(x=[X_news_train, X_tweet_train
				, X_comments_train
				], y=y_train, batch_size=40, epochs=1)
	# predictions = tCNN.predict([X_news_test, X_tweet_test, X_comments_test], batch_size=10)
	predictions = tCNN.predict([new_X_news_test, new_X_tweet_test
								, new_X_comments_test
								], batch_size=10)



	print('\nfinish fit\n')

	# predicted = np.argmax(predictions, axis=1)
	expected = np.argmax(y_test, axis=1)

	# ---------------------------------------------------- #
	# USE THIS ONLY IF THE FINAL DENSE LAYER SHAPE IS (#SAMPLES, 1)
	# print(np.unique(np.around(predictions)))
	# predictions = np.around(predictions)
	# predictions[predictions > 2] = 2
	# ---------------------------------------------------- #

	# print(tCNN.evaluate([X_news_test, X_tweet_test
	# 					 , X_comments_test
	# 					 ], y_test))
	print('Evaluate: ', tCNN.evaluate([new_X_news_test, new_X_tweet_test, new_X_comments_test], new_label))

	predicted = set_final_label(predictions, 1)

	print('Accuracy: ', accuracy_score(predicted, expected, normalize=True),
	'F1 score: ', f1_score(predicted, expected, average='weighted'))

	utils.plot_confusion_matrix(expected, predicted, label_encoder.classes_,
	output_file_path=None,
	normalize=True)

	# expected, predicted = new_label_func(X_test, predicted, expected)


	# print('Accuracy: ', accuracy_score(predicted, expected, normalize=True),
	# 	  'F1 score: ', f1_score(predicted, expected, average='weighted'))


if __name__ == '__main__':
    main()
