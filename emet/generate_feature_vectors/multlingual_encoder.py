import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import tf_sentencepiece

import sys


class MultilingualSentenceEncoder(object):

    __MODULE_URL = "https://tfhub.dev/google/universal-sentence-encoder-xling-many/1"
    __MODULE_URL_16LAN = "https://tfhub.dev/google/universal-sentence-encoder-multilingual-qa/1"
    __ELMO_URL = "https://tfhub.dev/google/Wiki-words-500-with-normalization/1"
    # __ELMO_URL = "https://tfhub.dev/google/universal-sentence-encoder/2"

    def __init__(self):

        self.embeddings = []

    def get_multilingual_embeddings(self, raw_documents):

        print('Vectorizing...')

        self.embeddings = []
        graph = tf.Graph()
        with graph.as_default():

            text_input = tf.placeholder(dtype=tf.string, shape=[None])
            # text_input = tf.compat.v1.placeholder(dtype=tf.string, shape=[None])
            # encoder = hub.Module(self.__MODULE_URL_16LAN)
            encoder = hub.Module("https://tfhub.dev/google/nnlm-en-dim128/1")
            # encoder = hub.Module(self.__MODULE_URL)
            embedded_text = encoder(text_input)
            init_op = tf.group([tf.global_variables_initializer(), tf.tables_initializer()])


        graph.finalize()

        session = tf.Session(graph=graph)
        session.run(init_op)

        for index, document in enumerate(raw_documents):
            embedding = session.run(embedded_text, feed_dict={text_input: [document]})
            self.embeddings.append(embedding)

            sys.stdout.write("\r {:<70}".format(index))
            sys.stdout.flush()

        # self.embeddings = raw_documents.apply(lambda tweet : session.run(embedded_text, feed_dict={text_input: tweet}))

        print('Saiu!')
        return self.embeddings
