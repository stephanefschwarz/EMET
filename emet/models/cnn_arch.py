from keras.layers import (concatenate, Flatten, AveragePooling1D,
                          Reshape, Conv1D, Dense, MaxPool1D,
                          Dropout, GlobalAveragePooling1D)

from keras import (Input, Model)

class TrueCNN:

    def __init__(self, news_input_shape=(1 ,1), tweet_input_shape=(1,1), comments_input_shape=None):

        print('init')

        self.model = self.__build_model(news_input_shape,
                                   tweet_input_shape,
                                   comments_input_shape)

    def __build_model(self, news_input_shape, tweet_input_shape, comments_input_shape):

        print('build model')

        news = Input(shape=news_input_shape, name='news_input')
        tweet = Input(shape=tweet_input_shape, name='tweets_input')
        # comments = Input(shape=comments_input_shape, name='comments_input')

        first_layer = Conv1D(filters=5, kernel_size=5, name='first_conv_news')(news)
        second_layer = Conv1D(filters=5, kernel_size=3, name='first_conv_tw')(tweet)
        # third_layer = Conv1D(filters=5, kernel_size=3, name='first_conv_com')(comments)

        first_layer = Flatten()(first_layer)
        second_layer = Flatten()(second_layer)
        # third_layer = Flatten()(third_layer)

        # if (comments_input_shape != None):
        # else:
        #     merged = concatenate([first_layer, second_layer])

        merged = concatenate([first_layer, second_layer])
        # merged = concatenate([first_layer, second_layer, third_layer])
        output = Dense(units=200, activation='relu', name='dense_layer_200')(merged)
        output = Reshape(target_shape=(200,1))(output)
        output = AveragePooling1D(pool_size=3, strides=1, name='first_avgPool')(output)
        output = Conv1D(filters=5, kernel_size=3, name='1_conv_conc')(output)
        output = Conv1D(filters=3, kernel_size=2, name='2_conv_conc')(output)
        output = Flatten()(output)
        output = Dropout(rate=0.5)(output)
        output = Dense(units=150, activation='relu', name="1_dense_layer")(output)
        output = Reshape(target_shape=(150, 1))(output)
        output = Conv1D(filters=5, kernel_size=2)(output)
        output = Flatten()(output)
        output = Dense(units=70, activation='relu')(output)
        # output = Dense(units=1, activation='relu', name="2_dense_layer")(output)
        output = Dense(units=3, activation='relu', name="2_dense_layer")(output)
        # if (comments_input_shape != None):
        # else:
        #     model = Model(inputs=[news, tweet], outputs=output)

        model = Model(inputs=[news, tweet
                             # , comments
                             ], outputs=output)

        # model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
        model.compile(optimizer='adadelta', loss='mean_squared_error', metrics=['accuracy'])

        return model
