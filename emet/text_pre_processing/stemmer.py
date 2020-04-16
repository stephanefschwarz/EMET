import nltk

from nltk.stem import LancasterStemmer, WordNetLemmatizer

def docment_stemming(document_tokens):
    """Stemming a list of words from a corpus of text.
    This library supports only english language.
    For every word the root (stem) of the word is returned.

    Parameters
    ----------
    document_tokens : pandas dataframe
        a list of each docment tokens
    Returns
    -------
    list<stem_tokens>
        the same list of tokens excepted that was stemmed
    """

    stemmer = LancasterStemmer()

    document_tokens.apply(lambda token: stemmer(token))

    return document_tokens

def docment_lemmatization(document_tokens):
    """Generate for each document token the lemma of
    the word. Reduces inflected words.

    Parameters
    ----------
    document_tokens : pandas dataframe
        a list of each docment tokens
    Returns
    -------
    list<tokens>
        the same list of tokens excepted that the inflected
        words was removed.
    """

    lemma = WordNetLemmatizer()

    document_tokens.apply(lambda token: lemma.lemmatize(token))

    return document_tokens

def remove_punctuation(document_tokens, punctuation_list=['.', ',', '?', ':', '!'.';']):
    """Removes punctuation from a list of tokens based
    on de passed punctuation list, or by default ['.', ',', '?', ':', '!'.';'].

    Parameters
    ----------
    document_tokens : pandas dataframe
        a list of each docment tokens
    punctuation_list : list<str>
        a list of all symbols to be removed from the passed list.
    Returns
    -------
    list<tokens>
        the same list of tokens excepted that the
        punctuation was removed.
    """

    for token in document_tokens:

        for punc in punctuation_list:

            document_tokens.remove(punc)

    return document_tokens
