import pandas as pd
import numpy as np

# event_start_with = ['underwater', 'sochi', 'varoufakis',
#                     'sandy', 'samurai', 'pigFish',
#                     'nepal', 'passport', 'malaysia',
#                     'livr', 'garissa', 'elephant',
#                     'eclipse', 'columbianChemicals', 'bringback'
#                     'boston', 'syrianboy']

# key_word_list = ['Underwater bedroom', 'Sochi Olympics',
#                  'Varoufakis and ZDF', 'Hurricane Sandy',
#                  'Samurai and Girl', 'pigFish',
#                  'Nepal earthquake', 'passport',
#                  'Malaysia flight 370', 'livr',
#                  'Garissa Attack', 'elephant rock',
#                  'Solar Eclipse', 'colombian chemicals',
#                  'Bring Back Our Girls', 'Boston Marathon bombing',
#                  'syrian boy beach']

# event_start_with = ['eclipse', 'garissa',
#                     'nepal', 'samurai',
#                     'syrianboy', 'varoufakis']
# # key_word_list = ['Garissa Attack', 'Nepal earthquake',
# #                  'Samurai and Girl', 'Solar Eclipse',
# #                  'Varoufakis and ZDF', 'syrian boy beach']
# key_word_list = ['Solar Eclipse', 'Garissa Attack',
#                  'Nepal earthquake', 'Samurai and Girl',
#                  'syrian boy beach', 'Varoufakis and ZDF']


key_word_list = ['Solar Eclipse', 'Garissa Attack',
                 'Nepal earthquake', 'Samurai and Girl',
                 'syrian boy beach', 'Varoufakis and ZDF',
                 'airstrikes','american soldier quran','ankara explosions',
                 'attacks paris','black lion','boko haram','bowie david',
                 'brussels car metro','brussels explosions','burst kfc',
                 'bush book','convoy explosion turkey','donald trump attacker',
                 'eagle kid','five headed snake','fuji','gandhi dancing',
                 'half everything','hubble telescope','immigrants','isis children',
                 'john guevara','McDonalds fee','nazi submarine','north korea',
                 'not afraid','pakistan explosion','pope francis','protest',
                 'refugees','rio moon','snowboard girl','soldier stealing',
                 'syrian children','ukrainian nazi','woman 14 children']

event_start_with = ['eclipse','garissa','nepal','samurai','syrianboy','varoufakis',
                    'airstrikes','american','ankara',
					'attacks','black','boko_haram','bowie_david',
					'brussels_car_metro','brussels_explosions','burst_kfc',
					'bush_book','convoy_explosion_turkey','donald_trump_attacker',
					'eagle_kid','five_headed_snake','fuji_lenticular','gandhi_dancing',
					'half_everything','hubble_telescope','immigrants','isis_children',
					'john_guevara','mc_donalds_fee','nazi_submarine','north_korea',
					'not_afraid','pakistan_explosion','pope_francis','protest',
					'refugees','rio_moon','snowboard_girl','soldier_stealing',
					'syrian_children','ukrainian_nazi','woman_14_children']



#
# bbc_news = pd.read_csv('./dataset/news_from_bbc_cleaned')
# tweets = pd.read_csv('./dataset/cleaned_twitter_posts.csv', engine='python')
bbc_news = pd.read_pickle('MediaEval_feature_extraction/dataset/ReutersNews.pkl')
tweets = pd.read_pickle('MediaEval_feature_extraction/dataset/test.pkl')

new_tweetId = []
new_tweetText = []
new_imageId = []
new_label = []
new_cleanedPost = []

new_news = []
new_newsKeyword = []
new_newsUrl = []

new_comments = []

for i, key in enumerate(event_start_with):
    print(i, " ------>> ", key)

    event = tweets['imageId(s)'].str.startswith(key)
    indexs = np.where(event)[0]

    event_news = bbc_news[bbc_news.key_word == key_word_list[i]]

    if (event_news.size == 0):

        continue

    for index in indexs:

        for ni, news in event_news.iterrows():
            try:

                new_tweetId.append(tweets['tweetId'].iloc[index])
                new_tweetText.append(tweets['tweetText'].iloc[index])
                new_imageId.append(tweets['imageId(s)'].iloc[index])
                new_label.append(tweets['label'].iloc[index])
                new_comments.append(tweets['comments'].iloc[index])
                # new_cleanedPost.append(tweets['cleaned_post'][index])

                new_news.append(news.news)
                new_newsKeyword.append(news.key_word)
                new_newsUrl.append(news.url)
            except:
                print(index)

final_dataset = pd.DataFrame(
                            {
                            'tweetId' : new_tweetId,
                            'tweetText' : new_tweetText,
                            'imageId' : new_imageId,
                            # 'cleaned_post' : new_cleanedPost,
                            'bbc_news' : new_news,
                            'key_word' : new_newsKeyword,
                            'url' : new_newsUrl,
                            'label' : new_label,
                            'comments' : new_comments
                            })

final_dataset.to_pickle('MediaEval_feature_extraction/dataset/testset_comments_newsReuters.pkl')
