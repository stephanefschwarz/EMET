import requests as r
from bs4 import BeautifulSoup as bs
import pandas as pd
import sys

class GetNewsFromReuters(object):

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

            page = r.get("https://www.reuters.com/search/news?sortBy=&dateRange=&blob="+str(word))

            soup = bs(page.content, 'html.parser')
            headers = soup.find_all('h3')

            for header in headers:

                url = header.contents[0].get('href')
                news_page = r.get("https://www.reuters.com/"+url)
                news_content = bs(news_page.content, 'html.parser')

                relevant_content = news_content.find('div', attrs={'class' : 'StandardArticleBody_body'})

                try:

                    items = relevant_content.find_all(['p'])

                    news = ""
                    for item in items:

                        news = news + " " + item.text

                    self.__news.append(news)
                    self.__urls.append("https://www.reuters.com/"+url)
                    self.__key_words.append(word)

                except Exception as e:

                    continue

            percentage = round((progress / len(key_words)) * 100, 2)
            output_print = "{}% | {}/{}".format(percentage, progress, len(key_words))
            sys.stdout.write("\r {:<70}".format(output_print))
            sys.stdout.flush()

        dic = {'news' : self.__news, 'url' : self.__urls, 'key_word' : self.__key_words}

        data = pd.DataFrame(dic)

        data.to_pickle('MediaEval_feature_extraction/dataset/testset_comments_newsReuters.pkl')


if __name__ == '__main__':

    key_word_list = ["Garissa Attack", "Nepal earthquake",
                     "Solar Eclipse", "Samurai and Girl",
                     "syrian boy beach", "Varoufakis and ZDF",
                     "airstrikes","american soldier quran","ankara explosions",
                     "attacks paris","black lion","boko haram","bowie david",
                     "brussels car metro","brussels explosions","burst kfc",
                     "bush book","convoy explosion turkey","donald trump attacker",
                     "eagle kid","five headed snake","fuji","gandhi dancing",
                     "half everything","hubble telescope","immigrants","isis children",
                     "john guevara","McDonalds fee","nazi submarine","north korea",
                     "not afraid","pakistan explosion","pope francis","protest",
                     "refugees","rio moon","snowboard girl","soldier stealing",
                     "syrian children","ukrainian nazi","woman 14 children"]



    # key_word_list = ["Garissa Attack", "Nepal earthquake",
    #                  "Solar Eclipse", "Samurai and Girl",
    #                  "syrian boy beach", "Varoufakis and ZDF"]

    module = GetNewsFromReuters()

    module.get_news(key_word_list)
