import sys
import pandas as pd
import requests
import nltk
nltk.download('stopwords')
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords

from bs4 import BeautifulSoup

# --- open dataset --- #
data = pd.read_csv('./dataset/translated_twitter_posts.csv')

documents = data['translated_posts']

# --- create an instance of tokenizer --- #

premises = []

tokenizer = RegexpTokenizer(r'\w+')

progress = 0
total_posts = documents.shape[0]

for document in documents:
    sentence = ''
    tokens = tokenizer.tokenize(document)
    for token in tokens:

        if not token in stopwords.words('english'):
            try:
                request = requests.get("http://www.urbandictionary.com/define.php?term={}".format(token))
                extract_mening = BeautifulSoup(request.content, 'html.parser')
                meaning = extract_mening.find("div",attrs={"class":"meaning"})
                if meaning != None:
                    meaning = meaning.text
                    sentence = sentence + meaning + ' '
                else:
                    sentence = sentence + token + ' '
            except Exception as e:
                print('Exception at token ', token, '\n', e)
        else:
            sentence = sentence + token + ' '

    premises.append(sentence)

    progress = progress + 1
    percentage = round((progress / total_posts) * 100, 2)

    output_print = "{}% | {}/{}".format(percentage, progress, total_posts)

    # Poor way to show a progress bar :|
    sys.stdout.write("\r {:<70}".format(output_print))
    sys.stdout.flush()

data['premises'] = premises
data.to_csv('./dataset/premises_twitter_posts.csv')
