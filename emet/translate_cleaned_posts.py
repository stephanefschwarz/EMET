from translate.translate_post import TranslateTexts

import pandas as pd

# --- open data --- #

data = pd.read_csv('./dataset/cleaned_twitter_posts.csv')

documents = data['cleaned_post']

# --- translate posts --- #

translator = TranslateTexts()

translator.translate_posts(documents, dest='en')

data['translated_posts'] = translator.translated_posts

data.to_csv('./dataset/translated_twitter_posts.csv')
