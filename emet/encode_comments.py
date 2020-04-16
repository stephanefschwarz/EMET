import pandas as pd

from generate_feature_vectors import multlingual_encoder

encoder = multlingual_encoder.MultilingualSentenceEncoder()

data = pd.read_pickle('./dataset/tweets_comments_and_bbc.pkl')

tweets_comments = []
for comments in data.comments:
    tweets_comments.append(' '.join(comments))

data['emb_comments'] = encoder.get_multilingual_embeddings(tweets_comments)

data.to_pickle('./dataset/tweets_comments_and_bbc.pkl')
