from regular_expressions import Patterns

import pandas as pd
import numpy as np

# --- open dataset --- #
data = pd.read_csv('./dataset/twitter_posts.csv')

# --- start pre processing data --- #

posts_processor = Patterns()

documents = data['tweetText']

pp_documents = posts_processor.pre_processing_text(documents)

data['cleaned_post'] = pp_documents

data.to_csv('./dataset/cleaned_twitter_posts.csv')
