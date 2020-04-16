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
	emoji_to_tag(document)
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


	def url_to_tag(self, document):
		"""Converts all present URLs in the document to a tag, in this case <URL>

		Parameters
		----------
		document : str
			a text to be converted
		Returns
		-------
		str
			the converted document
		"""    
		return re.sub(Patterns.URL, u'<URL>', document)

	def at_to_tag(self, document):
		"""Converts all present @ in the document to a tag, in this case <AT>

		Parameters
		----------
		document : str
			a text to be converted
		Returns
		-------
		str
			the converted document
		"""  
		return re.sub(Patterns.AT, u'<AT>', document)

	def hash_to_tag(self, document):
		"""Converts all present # in the document to a tag, in this case <HASH>

		Parameters
		----------
		document : str
			a text to be converted
		Returns
		-------
		str
			the converted document
		"""  
		return re.sub(Patterns.HASH, u'<HASH>', document)

	def number_to_tag(self, document):
		"""Converts all present numbers in the document to a tag, in this case <NUM>

		Parameters
		----------
		document : str
			a text to be converted
		Returns
		-------
		str
			the converted document
		"""  
		return re.sub(Patterns.NUM, u'<NUM>', document)

	def emoji_to_tag(self, document):
		"""Converts all present emoji in the document to a tag, in this case <EMO>

		Parameters
		----------
		document : str
			a text to be converted
		Returns
		-------
		str
			the converted document
		"""  
		return re.sub(Patterns.EMOJIS, u'<EMO>', document)

	def smileys_to_tag(self, document):
		"""Converts all present smileys in the document to a tag, in this case <SMI>

		Parameters
		----------
		document : str
			a text to be converted
		Returns
		-------
		str
			the converted document
		"""  
		return re.sub(Patterns.SMILEYS, u'<SMI>', document)

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

		for document in documents:

			document = self.emoji_to_tag(document)
			document = self.smileys_to_tag(document)
			document = self.url_to_tag(document)
			document = self.at_to_tag(document)
			document = self.hash_to_tag(document)
			document = self.number_to_tag(document)

			processed_documents.append(document)

		return processed_documents


if __name__ == '__main__':

	texts = ['#ThisIsATest', '@userRef represents a user reference', '1000 Years - Music', 'I love Ice Cream :-) :)', 'Take a loke at: https://bitbucket.org/StephaneSchwarz']

	p = Patterns()

	all_processing = p.pre_processing_text(texts)

	print(all_processing)

	for text in texts:

		only_hash = p.hash_to_tag(text)
		print(only_hash)

