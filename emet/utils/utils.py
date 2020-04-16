# ---------- PACKAGE IMPORTATION ---------- #

from sklearn.manifold import TSNE
import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
sns.set_style('darkgrid')
sns.set_palette('muted')
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels
import matplotlib.pyplot as plt
import pickle

def plot_tsne(documents, labels,  output_file_path, n_components=2, verbose=1, random_state=0, n_iter=1000):

	tsne = TSNE(n_components=n_components, verbose=verbose, random_state=random_state, n_iter=n_iter)

	# If is a sparce matrix converts to a dense
	if (type(documents) == 'scipy.sparse.csr.csr_matrix'):

		documents = documents.todense()

	clusters = tsne.fit_transform(documents)

	show_tsne(clusters, labels, output_file_path)


# This code was based on https://www.datacamp.com/community/tutorials/introduction-t-sne
def show_tsne(clusters, str_labels):
    print(str_labels.shape)
    labels = pd.Categorical(pd.factorize(str_labels)[0])
    n_classes = len(np.unique(labels))
    if(len(np.unique(labels)) > 6):
        palette = np.array(sns.color_palette('colorblind', n_classes))
    else:
        flatui = ["#9b59b6", "#3498db", "#34495e", "#e74c3c", "#2ecc71", "#95a5a6"]
        palette = np.array(sns.color_palette(flatui, n_classes))
    fig = plt.figure(figsize=(10, 10))
    aux = plt.subplot(aspect='equal')

    sc = aux.scatter(clusters[:, 0], clusters[:, 1], lw=0, s=40, c=palette[labels.astype(np.int8)])

    plt.xlim(-25, 25)
    plt.ylim(-25, 25)
    aux.axis('off')
    aux.axis('tight')

    plt.savefig(output_file_path)

    return


def save_most_frequent_words(n_words, tfidf_dtm, feature_names):

	word_frequencies = np.sum(tfidf_dtm, axis=0)
	# the highest to the lowest

	sorted_words = np.flip(np.argsort(word_frequencies), axis=1)
	words = []
	frequencies = []

	for i in range(0, n_words):

		index = sorted_words[0, i]

		words.append(feature_names[index])

		frequencies.append(word_frequencies[0, index])

	plt.hlines(y=words, xmin=0, xmax=frequencies*100,color='skyblue')
	plt.plot(frequencies, words, 'o')

	title = "{} most common words".format(n_words)
	plt.title(title)
	plt.xlabel('Frequency')
	plt.yticks(rotation=25)
	plt.ylabel('Word')

	file_output_path = title + '.png'
	plt.savefig(file_output_path, bbox_inches='tight',dpi=100)

def plot_confusion_matrix(y_true, y_pred, classes,
					  output_file_path,
                      normalize=False,
                      title=None,
                      cmap=plt.cm.Oranges):
	"""
	This function prints and plots the confusion matrix.
	Normalization can be applied by setting `normalize=True`.
	"""
	if not title:
		if normalize:
			title = 'Normalized confusion matrix'
		else:
			title = 'Confusion matrix, without normalization'

	# Compute confusion matrix
	cm = confusion_matrix(y_true, y_pred)
	# Only use the labels that appear in the data
	print(cm)
	exit()
	classes = classes[unique_labels(y_true, y_pred)]
	if normalize:
		cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
		print(title)
	else:
		print('Confusion matrix, without normalization')

	fig, ax = plt.subplots()
	im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
	ax.figure.colorbar(im, ax=ax)
	# We want to show all ticks...
	ax.set(xticks=np.arange(cm.shape[1]),
	   yticks=np.arange(cm.shape[0]),
	   # ... and label them with the respective list entries
	   xticklabels=classes, yticklabels=classes,
	   title=title,
	   ylabel='True label',
	   xlabel='Predicted label')

	# Rotate the tick labels and set their alignment.
	plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
	     rotation_mode="anchor")

	# Loop over data dimensions and create text annotations.
	fmt = '.2f' if normalize else 'd'
	thresh = cm.max() / 2.
	for i in range(cm.shape[0]):
		for j in range(cm.shape[1]):
			ax.text(j, i, format(cm[i, j], fmt),
	                    ha="center", va="center",
	                    color="white" if cm[i, j] > thresh else "black")
	fig.tight_layout()

	plt.savefig(output_file_path)


def write_report(output_file_path, acc, f1, history, evaluate):
	results = {
				'test_acc: ' : acc,
				'test_f1: ' : f1,
				'evaluate' : evaluate,
				'train_acc' : history['accuracy'],
				'train_loss' : history['loss']
				}

	f = open(output_file_path, 'wb')
	pickle.dump(results, f)
	f.close()

	return
