# ---------- PACKAGE IMPORTATION ---------- #
import sklearn
import tensorflow
# ---------- MODELS ---------- #
from sklearn.naive_bayes import GaussianNB as NB
from sklearn.ensemble import RandomForestClassifier as RF
from sklearn.neural_network import MLPClassifier as MLP
# ---------- METRICS ---------- #
from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import train_test_split

class Models(object):

    def __init__(self, model):

        self.X = None
        self.y = None
        self._X_val = None
        self._y_val = None
        self._y_train = None
        self._X_train = None
        self._prediction = None

    @property
    def get_X_val(self):
        return self._X_val
    @property
    def get_X_train(self):
        return self._X_train
    @property
    def get_y_val(self):
        return self._y_val
    @property
    def get_y_train(self):
        return self._y_train
    @property
    def get_prediction(self):
        return self._prediction

    def __split_dataset(self):
        """Splits the dataset into train and validation set of data

		Parameters
		----------
        Uses only the class attributes.
		Returns
		-------
        Store the result on the class attributes also.
		"""

        self.X_train, self.X_val, \
        self.y_train, self.y_val = \
        train_test_split(self.X, self.y, test_size=0.3, random_state=42)


    def random_forest(self, n_estimators=38, n_jobs=6):
        """Random Forest Classifier

		Parameters
		----------
		n_estimators : int
			number o trees in the florest
		n_jobs : int
			number of jobs to run in parallel.
		Returns
		-------
		Store the predictions on the _prediction attribute.
		"""

        model = RF(n_estimators=n_estimators, n_jobs=n_jobs)
        model.fit(self._X_train, self._y_train)

        self._prediction = model.predict(self._X_val, self._y_val)

    def naive_bayes(self):
        """Compute predictions on naive bayes algorithm.

		Parameters
		----------
		Use just class attributes
		Returns
		-------
		Store the predictions on the _prediction attribute.
		"""

        model = NB()
        model.fit(self._X_train, self._y_train)

        self._prediction = model.predict(self._X_val, self._y_val)

    def multilayer_perceptron(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5,2), random_state=1):
        """Compute predictions based on multilayer perceptron algorithm.

		Parameters
		----------
		solver : {‘lbfgs’, ‘sgd’, ‘adam’}, default ‘adam’
            Weight optimization
        alpa : float
            regularization term
        hidden_layer_sizes : tuple
            number of neurons in the hidden layer
        random_state : int
            seed used by the random number generator.
		-------
		Store the predictions on the _prediction attribute.
		"""

        model = MLP(solver=solver, alpha=alpha, \
        hidden_layer_sizes=hidden_layer_sizes, random_state=random_state)
        model.fit(self._X_train, self._y_train)

        self._prediction = model.predict(self._X_val, self._y_val)

    def get_f1score_and_accuracy(self):
        """Computes model score and accuracy.

		Parameters
		----------
		Uses only class attribute
		-------
		f1_score, accuracy
            returns the f1_score and accuracy of the predicted values
            from a model.
		"""

        score = f1_score(self._y_val, self.prediction, average='weighted')
        accuracy = accuracy_score(self._y_val, self.prediction, normalized=True)

        return score, accuracy

    def nn_tensorflow(x_train, y_train, x_val, y_val):

        x = tensorflow.placeholder(dtype=tensorflow.float32, shape=[None, 500])
        y = tensorflow.placeholder(dtype=tensorflow.int32, shape=[None])

        flatten = tensorflow.contrib.layers.flatten(x)

        fully_cone = tensorflow.contrib.layers.fully_connected(flatten, 62, tensorflow.nn.relu)

        loss = tensorflow.reduce_mean(tensorflow.nn.sparse_softmax_cross_entropy_with_logits(
        labels=y, logits=fully_cone))

        train_op = tensorflow.train.AdamOptimizer(learning_rate=0.01).minimize(loss)

        correct_pred = tensorflow.argmax(fully_cone, 1)

        acc = tensorflow.reduce_mean(tensorflow.cast(correct_pred, tensorflow.float32))

        tensorflow.set_random_seed(0)

        sess = tensorflow.Session()
        sess.run(tensorflow.global_variables_initializer())

        for i in range(201):

            print('EPOCH',i)
            _, acc_val = sess.run([train_op, acc],
                                  feed_dict={x: x_train, y: y_train})

            if (i % 10 == 0):
                print('Loss: ',loss)
            print('EPOCH DONE')

        self._prediction = sess.run([correct_pred], feed_dict={x:x_val})[0]

        return
