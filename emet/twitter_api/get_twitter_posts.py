# ---------- PACKAGE IMPORTATIONS ---------- #

import twitter
import pandas as pd
import numpy as np
import time
import json
import argparse
import logging
logging.basicConfig(filename='app.log', filemode='w', level=logging.INFO, format='%(name)s - %(levelname)s - %(message)s')


# ---------- SINGLETON CLASS FOR TWITTER APP ---------- #

class TwitterApp:
	"""
	A class used to generats an instance for Twitter API.

	Attributes
	----------
	None
	
	Methods
	-------
	get_twitter_app_instance():
		returns the instance for twitter API to be used on the application
	"""
	
	# The only Twitter app instance that will exist.
	__api = None

	def __init__(self, credential):
		"""Build the instance in the constructor.

		Parameters
		----------
		self : self
			Instace to itself.
		credential : dict
			credential to twitter app
		Returns
		-------
		API
			an instance to the Twitter API.
		"""

		__CONSUMER_KEY = credential['twitter_credentials']['CONSUMER_KEY']
		__CONSUMER_SECRET = credential['twitter_credentials']['CONSUMER_SECRET']
		__ACCESS_TOKEN = credential['twitter_credentials']['ACCESS_TOKEN']
		__ACCESS_SECRET = credential['twitter_credentials']['ACCESS_SECRET']
		
		if TwitterApp.__api == None:

			TwitterApp.__api = twitter.Api(consumer_key = __CONSUMER_KEY,
							  consumer_secret = __CONSUMER_SECRET,
							  access_token_key = __ACCESS_TOKEN,
					  		  access_token_secret = __ACCESS_SECRET)

			try:
				
				TwitterApp.__api.VerifyCredentials()
				print("Authenticated!")

			except Exception as e:
				
				print("Could not authenticate you, verify credentials.")
				
	def get_twitter_app_instance(self):
		"""Get the instance for the Twitter app

		Parameters
		----------
		self : self
			Instace to itself.
		Returns
		-------
		API
			An instance to the Twitter app.
		"""

		if TwitterApp.__api == None:

			TwitterApp(credential);
		
		return TwitterApp.__api


class FindTweetsByID(TwitterApp):
	"""
	A class used to find tweets on the twitter API through IDs

	Attributes
	----------
	None
	
	Methods
	-------
	get_tweets(self, output_path, tweets_ids)
	output_path : str
		the location to store the obtained tweets
	tweets_ids : list
		a list of tweets IDs

	"""
	# ---------- CONSTANTS ---------- #

	MAX_TWEETS_PER_REQUEST = 100 # Max value allowed by Twitter API

	# ---------- FUNCTIONS DEFINITION ---------- #

	def get_tweets(self, output_path, tweets_ids):
		"""Get tweets from IDs

		Parameters
		----------
		output_path : str
			the location to store the obtained tweets
		tweets_ids : list
			a list of tweets IDs
		Returns
		-------
		list
			a list of tweets and your IDs, if the ID is not known, nothing is returned.
		"""

		loading = 0

		app = TwitterApp.get_twitter_app_instance(self)

		tweets_content = []

		new_tweets_ids = []

		qty_tweets = len(tweets_ids)

		last_index = 0

		while True:
			
			try:

				response = app.GetStatuses(tweets_ids[last_index:last_index+100], map=True)
				
			except Exception as e:

				# save the available posts to this time
				dataset = pd.DataFrame({'tweet_id':new_tweets_ids, 'post_content':tweets_content})
				write_tweets(output_path, dataset)

				logging.info(''.join(['Error on request ', str(loading)]))

				print("ERROR:", e)

				'''
				Usually, the rate limit allowed by Twitter API is exceeded (in this case GET statuses/lookup is 900 for user auth and 300 for the app auth for every 15 minutes), one way to deal with it is sleeping the code for approximately 15 minutes to continue after.
				'''
				time.sleep(950)

				try:

					response = app.GetStatuses(tweets_ids[last_index:last_index+100], map=True)
				
				except Exception as e:

					print(e)
					exit(1)


			for id_value, text in response.items():			

				# This means that the post is not available now.
				if (text == None):
					continue

				else:

					new_tweets_ids.append(id_value)
					tweets_content.append(text.text)

			# Each request gets 100 posts
			last_index = last_index + 100

			# There is no more IDs
			if (last_index > qty_tweets):
				break	
		
		# save all tweets
		dataset = pd.DataFrame({'tweet_id':new_tweets_ids, 'post_content':tweets_content})
		write_tweets(output_path, dataset)
		

# ---------- FILE METHODS ---------- #

def read_text_file(file_path):

	logging.info("Reading file.")

	"""Read file from path

		Parameters
		----------
		file_path : str
			file path to the list of IDs to be found
		
		Returns
		-------
		list
			List of IDs
	"""	

	with open(file_path) as file:

		texts = file.read().splitlines()

	file.close()
	
	return texts

def write_tweets(output_path, tweets_dataframe):

	logging.info("Writing result.")

	"""Write the generated file

		Parameters
		----------
		output_path : str
			The path to write the output file
		tweets_dataframe : Pandas DataFrame		
			a dataframe to be stored
		Returns
		-------
		None
	"""
	tweets_dataframe.to_csv(output_path)

	return

def command_line_parsing():
	"""Parse command lines

		Parameters
		----------
		file_path : str
			file path to the list of tweets IDs
		output_file_path : str
			file to store the output file
		file_credential : str
			file to get the twitter app credentials

		Returns
		-------
		parser
			The arguments from command line
	"""	
	logging.info("Parsing command line")
	
	parser = argparse.ArgumentParser(description = __doc__)

	parser.add_argument('--file-path', '-f', 
						dest='file_path', 
						required=True,
						help='File path to the list of tweets IDs.')

	parser.add_argument('--file-credential', '-c', 
						dest='file_credential', 
						required=True,
						help='File with the twitter app credentials.')

	parser.add_argument('--output-file-path', '-o', 
						dest='output_file_path', 
						required=True,
						help='Output path to store tweets.')


	return parser.parse_args()

	

if __name__ == '__main__':


	args = command_line_parsing()

	with open(args.file_credential) as file:

		credentials = json.load(file)
	
	tweets_ids = read_text_file(args.file_path)
	
	twitter = FindTweetsByID(credentials)

	twitter.get_tweets(args.output_file_path, tweets_ids)

