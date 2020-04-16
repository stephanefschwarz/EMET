import requests as r
from bs4 import BeautifulSoup as bs
import pandas as pd
import sys

class GetNewsFromBBC(object):

    __news = []
    __key_words = []
    __urls = []

    def get_news(self, key_words):
        """Get news from bbc based on a list of key words

		Parameters
		----------
		key_words : list
            A list of key words to search on bbc

		Returns
		-------
		list
            a list of news that was obtained using the passed key words
		"""
        progress = 0
        for word in key_words:
            progress = progress + 1

            page = r.get("https://www.bbc.co.uk/search?q="+str(word))

            soup = bs(page.content, 'html.parser')
            headers = soup.find_all('h1')

            for header in headers:

                url = header.contents[0].get('href')
                news_page = r.get(url)
                news_content = bs(news_page.content, 'html.parser')

                relevant_content = news_content.find('div', attrs={'class' : 'story-body__inner'})

                try:

                    items = relevant_content.find_all(['p', 'h2'])

                    news = ""
                    for item in items:

                        news = news + " " + item.text

                    self.__news.append(news)
                    self.__urls.append(url)
                    self.__key_words.append(word)

                except Exception as e:

                    continue

            percentage = round((progress / 14) * 100, 2)
            output_print = "{}% | {}/{}".format(percentage, progress, 14)
            sys.stdout.write("\r {:<70}".format(output_print))
            sys.stdout.flush()

        dic = {'news' : self.__news, 'url' : self.__urls, 'key_word' : self.__key_words}

        data = pd.DataFrame(dic)

        data.to_csv('./dataset/news_from_bbc.csv')


if __name__ == '__main__':

    key_word_list = ["Hurricane Sandy", "Boston Marathon bombing",
                     "Sochi Olympics", "Bring Back Our Girls",
                     "Malaysia flight 370", "colombian chemicals",
                     "elephant rock", "Underwater bedroom",
                     "Nepal earthquake", "Solar Eclipse",
                     "Garissa Attack", "Samurai and Girl",
                     "syrian boy beach", "Varoufakis and ZDF"]

    module = GetNewsFromBBC()

    module.get_news(key_word_list)
