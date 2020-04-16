# ---------- PACKAGE IMPORTATION ---------- #
from sklearn.feature_extraction.text import TfidfVectorizer

from utils import plot 

def tfidf_verctorization(text_list, analyzer='word', ngram_range=(2,2), stop_words='english', lowercase=True, max_features=500, use_idf=True):
	"""Generate TF-IDF features vectors from a list of text.

	Parameters
	----------
	text_lis : list
		a list of text to extract the tfidf vectors
	analyzer : str
		the kind of analyzer to be considerated 'word' or 'char'
	ngram_range : tuple
		number of gram that will be considerated
	stop_words : str
		the language to remove stop words
	lowercase : boolean
		if lowercase will be considerated, by default True
	max_features : int
		feature vector length
	use_idf : boolean
		if will use idf value, by default True

	Returns
	-------
	list
		a list TF-IDF vector values.
	"""

	tfidf = TfidfVectorizer(analyzer=analyzer, ngram_range=ngram_range, stop_words=stop_words, lowercase=lowercase, max_features=max_features, use_idf=use_idf)

	feature_vector = tfidf.fit_transform(text_list)
	
	plot.save_most_frequent_words(len(tfidf.get_feature_names())-1, feature_vector, tfidf.get_feature_names())


	return feature_vector

# if __name__ == '__main__':
	
	# texts_to_test = ['Esse é um teste', 'Mais um texto para test de conceito', 'Isso é necessário para verificação dos textos']


	
	# import numpy as np
	# print(np.sum(tfidf_verctorization(texts_to_test), axis=0))
	# print('\n\n')
	# print(tfidf_verctorization(texts_to_test))