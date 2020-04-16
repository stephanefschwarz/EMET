# --- PACKAGE IMPORTATION ---

import re

class Patterns:

	"""
	A class used to replace some social media patterns into a tag.

	Attributes
	----------
	None

	Methods
	-------
	url_to_tag(document)
		converts URLs to a tag
	at_to_tag(document)
		converts @ to a tag
	hash_to_tag(document)
		converts # to a tag
	number_to_tag(document)
		converts numbers to a tag
	emoji_to_tag(document) or remove_emoji_pattern(document)
		converts emojis to a tag
	smileys_to_tag(document)
		converts smileys to a tag
	pre_processing_text(documents)
		execute all the above method for each document in a list of documents
	"""

	URL = re.compile('((([A-Za-z]{3,9}:(?:\/\/)?)(?:[\-;:&=\+\$,\w]+@)?[A-Za-z0-9\.\-]+|(?:www\.|[\-;:&=\+\$,\w]+@)[A-Za-z0-9\.\-]+)((?:\/[\+~%\/\.\w\-_]*)?\??(?:[\-\+=&;%@\.\w_]*)#?(?:[\.\!\/\\\w]*))?)')

	AT = re.compile('@\w+')

	HASH = re.compile('#\w+')

	NUM = re.compile('\d+')

	SMILEYS = re.compile(r'(?:X|:|;|=)(?:-)?(?:\)|\(|O|D|P|S){1,}', re.IGNORECASE)

	EMOJIS = re.compile("["
        u"\U0001F600-\U0001F64F"
        u"\U0001F300-\U0001F5FF"
        u"\U0001F680-\U0001F6FF"
        u"\U0001F1E0-\U0001F1FF"
        u"\U00002600-\U000027BF"
        u"\U0001f300-\U0001f64F"
        u"\U0001f680-\U0001f6FF"
        u"\u2600-\u27BF"
        u"\uD83C"
        u"\uDF00-\uDFFF"
        u"\uD83D"
        u"\uDC00-\uDE4F"
        u"\uD83D"
        u"\uDE80-\uDEFF"
        "]+", flags=re.UNICODE)


	def url_to_tag(self, document, icon='<URL>'):
		"""Converts all present URLs in the document to a tag, in this case <URL>

		Parameters
		----------
		document : str
			a text to be converted
		icon : str
			the icon that will be replaced at the RE value
			valid options '', <URL>
		Returns
		-------
		str
			the converted document
		"""
		return re.sub(Patterns.URL, icon, document)

	def at_to_tag(self, document, icon='<AT>'):
		"""Converts all present @ in the document to a tag, in this case <AT>

		Parameters
		----------
		document : str
			a text to be converted
		icon : str
			the icon that will be replaced at the RE value
			valid options '', <AT>
		Returns
		-------
		str
			the converted document
		"""
		return re.sub(Patterns.AT, icon, document)

	def hash_to_tag(self, document, icon='<HASH>', keep_words=False):
		"""Converts all present # in the document to a tag, in this case <HASH>

		Parameters
		----------
		document : str
			a text to be converted
		icon : str
			the icon that will be replaced at the RE value
			valid options '', <HASH>
		keep_words : boolean
			if the words after the #icon will continue

		Returns
		-------
		str
			the converted document
		"""
		if(keep_words):
			return re.sub('#', '', document)

		return re.sub(Patterns.HASH, icon, document)

	def number_to_tag(self, document, icon='<NUM>'):
		"""Converts all present numbers in the document to a tag, in this case <NUM>

		Parameters
		----------
		document : str
			a text to be converted
		icon : str
			the icon that will be replaced at the RE value
			valid options '', <NUM>
		Returns
		-------
		str
			the converted document
		"""
		return re.sub(Patterns.NUM, icon, document)

	def emoji_to_tag(self, document, icon='<EMO>'):
		"""Converts all present emoji in the document to a tag, in this case <EMO>

		Parameters
		----------
		document : str
			a text to be converted
		icon : str
			the icon that will be replaced at the RE value
			valid options '', <EMO>
		Returns
		-------
		str
			the converted document
		"""
		return re.sub(Patterns.EMOJIS, icon, document)

	def smileys_to_tag(self, document, icon='SMI'):
		"""Converts all present smileys in the document to a tag, in this case <SMI>

		Parameters
		----------
		document : str
			a text to be converted
		icon : str
			the icon that will be replaced at the RE value
			valid options '', <SMI>
		Returns
		-------
		str
			the converted document
		"""
		return re.sub(Patterns.SMILEYS, icon, document)

	def pre_processing_text(self, documents):
		"""Executes all pre-processing codes to converts social media Patters into a tag.

		Parameters
		----------
		documents : list
			a list of documents
		Returns
		-------
		list
			the converted documents
		"""
		processed_documents = []

		progress = 0
		total_posts = len(documents)

		for document in documents:

			document = self.remove_emoji_pattern(document)
			# document = self.emoji_to_tag(document, icon='')
			document = self.smileys_to_tag(document)
			document = self.url_to_tag(document)
			document = self.at_to_tag(document)
			document = self.hash_to_tag(document)
			# document = self.number_to_tag(document, icon='')

			processed_documents.append(document)

			progress = progress + 1

			percentage = round((progress / total_posts) * 100, 2)
			output_print = "{}% | {}/{}".format(percentage, progress, total_posts)

			# Poor way to show a progress bar :|
			sys.stdout.write("\r {:<70}".format(output_print))
			sys.stdout.flush()

		return processed_documents

	def remove_emoji_pattern(self, document):
		"""Removes all emoji pattern of the sentence based on the ascii code

		Parameters
		----------
		document: str
			a single sentence
		Returns
		-------
			the sentence without the emoji pattern
		"""
		return document.encode('ascii', 'ignore').decode('ascii')

	def add_space_to_camelCase(self, document):
		"""Add spacs into camel case sentences

		Parameters
		----------
		document: str
			a single sentence
		Returns
		-------
			the sentence with spaces
		"""

		return re.sub('([A-Z][a-z]+)', r' \1', re.sub('([A-Z]+)', r'\1', document))

def remove_retweet(dataset):

	return dataset[~dataset.tweetText.str.startswith('RT')]

if __name__ == '__main__':

	texts = ['This is a @user test aiming to #Know the :-) powerfull of my regular http://t.co/v82rdCVIqH FAST Mittag ▶ Riesen',
	'#ThisIsATest',
	'@userRef represents a user reference',
	'1000 Years - Music',
	'I love Ice Cream :-) :)',
	'Take a loke at: https://bitbucket.org/StephaneSchwarz']

	p = Patterns()

	all_processing = p.pre_processing_text(texts)

	print(all_processing)

	for text in texts:

		only_hash = p.hash_to_tag(text)
		# print(only_hash)
