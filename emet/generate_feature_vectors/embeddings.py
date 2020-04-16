# ---------- PACKAGE IMPORTATION ---------- #
import gensim
import nltk
import numpy
import os

from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from gensim.models import Word2Vec as w2v
from gesim.utils import simple_preprocessing

class EmbeddingWord2Vec(object):
	"""
	A class used to generate Word2Vec embeddings

	Attributes
	----------
	__model:
		pre-trained model for Word2Vec
	documents:
		a set of text documents
	tokens:
		a list of document tokens
	embeddings:
		document embeddings
	Methods
	-------
	__tokenize_documents(self)
		tokenize a document, for instance, "This sentence will be converted to"
		["This", "sentence", "will", "be", "converted", "to"] a list of tokens.
	__remove_stop_words(self, language='english')
		PRIVATE METHOD to remove all stopwords besed on the selected language (default english)
	generate_embeddings(generate_missing=False, dim=300)
		generate embeddings from document tokens
	__get_per_document_embeddings(document, generate_missing, dim):
		PRIVATE AND INNER METHOD to help with generate_embeddings function
	train_embedding(feature_dim, min_count, context_size, seed, downsampling, save_model, model_name)
		traine from scratch word embeddings for a document
	"""

	def __init__(self, model_file_path=None, documents, c_format=True):
		"""Initialize some variables

		Parameters
		----------
		model_file_path : str
			the path to the pre-trained model file
		documents: list<str>
			a list of documents
		c_format : boolean
			if will load as the original C tool
		Returns
		-------
		None
		"""
		if (model_file_path == None):
			self.__model = None
		elif (c_format == True):
			self.__model = w2v.load_word2vec_format(model_file_path, binary=True)
		else:
			self.__model = w2v.load(model_file_path)

		self.documents = documents
		self.tokens = None
		self.embeddings = None

	def __tokenize_documents(self, remove_stop_words=True, language='english'):
		"""Converts all documents in a list of tokens per document.

		Parameters
		----------
		remove_stop_words : boolean
			a flag to indicate if the stopwords should be removed
		language:
			stopwords language pattern
		Returns
		-------
		list<tokens>
			a list of tokens
		"""

		# Splits the texts with a regular expression in this case all occurrences of a word.
		tokenizer = RegexpTokenizer(r'\w+')

		if (type(documents) == 'list'):

			self.tokens = [tokenizer.tokenize(document) for document in self.documents]
		# If is not a list probably is a pandas dataframe
		else:

			self.tokens = documents.apply(tokenizer.tokenize)

		if (remove_stop_words == True):

			__remove_stop_words(language)

		return

	def __remove_stop_words(self, language='english'):
		"""Remove all stopwords from the tokenized list of documents

		Parameters
		----------
		language : str
			the language of the stopwords
		Returns
		-------
		list<tokens>
			a list of document tokens without stopwords
		"""

		stopwords = set(stopwords.words(language))
		self.tokens = self.tokens.apply(lambda word_token: [word for word in word_token if word not in stopwords])

		return

	def generate_embeddings(generate_missing=False, dim=300):
		"""Generate form a list of tokens the document embedding besed on a
		pre-trained model.

		Parameters
		----------
		generate_missing : boolean
			this variable indicates if a word is not in the pre-trained corpus
			what will be your value. If False complite with zeros, otherwise
			random values.
		Returns
		-------
		list
			a list of document embeddings
		"""

		self.embeddings = self.tokens.apply(lambda document: __get_per_document_embeddings(document, generate_missing, dim))

		def __get_per_document_embeddings(document, generate_missing, dim):

			# Basically if a word from the tokenized document list
			# is present in the pre-trained word embeddings we get
			# this value, but if not, there are two options:
			# 1. generate it for random or 2. set all the n-dim as zero.
			if (generate_missing):

				document_embedding = [self.__model[word] if word in self.__model else np.zeros(dim) for word in document]
			else:

				document_embedding = [self.__model[word] if word in self.__model else np.random.rand(dim) for word in document]

			# Now we need a vector from a document not for all the words,
			# so we average the obtained value of every word in a document.
			# v1[0, 1, 2, ..., 299] + v2[0, 1, 2, ..., 299] + ... + vn[0, 1, 2, ..., 299] = V[0, 1, 2, ..., 299]
			# after sum, we average by the length of the document, in other words number of words.
			length = len(document_embedding)
			document_embedding = numpy.sum(document_embedding, axis=0)
			document_embedding = numpy.divide(document_embedding, length)

			return document_embedding

		def train_embedding(feature_dim=300, min_count=1, context_size=7, seed=1, downsampling=1e-3, save_model=True, model_name="word2vec_embeddings.w2v"):
			"""Train embeddings from scratch.

			Parameters
			----------
			feature_dim : int
				dimensionality of the word vectors
			min_count : int
				ignores all word with frequency lower than this value
			context_size : int
				max. distance between the current word and predicted within a
				sentence
			seed : int
				seed for the random number generator, needed to be reproducible
			downsampling : int
				the threshold for configuring which higher-frequency words are
				randomly downsampled
			save_model : boolean
				if true saves the trained model
			model_name : str
				model file name, required if save_model is equal to True

			Returns
			-------
			list
				a list of document embeddings
			"""
			self.__model = w2v.Word2Vec(
			size=feature_dim,
			seed=seed,
			sg=1,
			min_count=min_count,
			window=context_size,
			sample=downsampling
			)

			self.__model.build_vocab(self.tokens)

			self.__model.train(self.tokens)

			if (save_model):
				folder_name = "trained_models"

				if not os.path.exists(folder_name):
					os.makedirs(folder_name)
				self.__model.save(os.path.join(folder_name, model_name))

			return
