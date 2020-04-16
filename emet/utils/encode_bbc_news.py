import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import tf_sentencepiece
import pandas as pd
import sys

dataset = pd.read_csv('./dataset/tweets_and_bbcNews.csv')

module_url = "https://tfhub.dev/google/universal-sentence-encoder-xling-many/1"

g = tf.Graph()
with g.as_default():
    text_input = tf.placeholder(dtype=tf.string, shape=[None])
    xling_8_embed = hub.Module(module_url)
    embedded_text = xling_8_embed(text_input)
    init_op = tf.group([tf.global_variables_initializer(), tf.tables_initializer()])

g.finalize()

session = tf.Session(graph=g)
session.run(init_op)

new_feature_vector = []

for index, document in enumerate(dataset.bbc_news):

    embedding = session.run(embedded_text, feed_dict={text_input: [document]})
    new_feature_vector.append(embedding)

    sys.stdout.write("\r {:<70}".format(index))
    sys.stdout.flush()

tweets_feature_vector = []

for index, document in enumerate(dataset.cleaned_post):

    embedding = session.run(embedded_text, feed_dict={text_input: [document]})
    tweets_feature_vector.append(embedding)

    sys.stdout.write("\r {:<70}".format(index))
    sys.stdout.flush()


to_save = pd.DataFrame({'news_featVec':new_feature_vector,
                       'tweet_featVec':tweets_feature_vector})

new_dataset = dataset + to_save

new_dataset.to_pickle('./dataset/embedded_tweets_and_bbcNews.pkl')
