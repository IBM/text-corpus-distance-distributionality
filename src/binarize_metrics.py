from compcor.corpus_metrics import *
from sklearn.svm import SVC
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from collections import namedtuple
from operator import itemgetter

import numpy as np
from numpy import random
import prdc.prdc as pr
PR = namedtuple('pr', 'precision recall distance')
DC = namedtuple('dc', 'density coverage distance')
import pandas as pd
import seaborn as sns



#
# def equal_element_pair_indices(lst):
# 	# all elements are equal to themselves
# 	n = len(lst)
# 	first_idx = list(range(n))
# 	second_idx = list(range(n))
# 	# now compare others
# 	for ii, item1 in enumerate(lst[:-1]):
# 		for jj in range(len(lst)):
# 			if item1 == lst[jj]:
# 				# both orders
# 				first_idx.extend([ii,jj])
# 				second_idx.extend([jj, ii])
# 	# return in format that can be used to index directly
# 	return (np.array(first_idx).astype(int), np.array(second_idx).astype(int))

class LogisticRegPred:
	# logistic regression where the predict gives the probability, not the class itself
	def __init__(self, balanced=True):
		self.clf = LogisticRegression(class_weight='balanced' if balanced else None)

	def fit(self, X, y):
		self.clf.fit(X, y)

	def predict(self, X):
		return self.clf.predict_proba(X)[:, 0]


class SVMPred:
	# SVM prediction
	def __init__(self, balanced=True):
		self.clf = SVC(class_weight='balanced' if balanced else None, probability=True)

	def fit(self, X, y):
		self.clf.fit(X, y)

	def predict(self, X):
		return self.clf.predict_proba(X)[:, 0]


# for non-semantic, assign 0 distance if exactly equal, otherwise 1
class NonsemanticPred:
	# all placeholders
	def __init__(self):
		self.clf = None

	def fit(self, X, y):
		pass

	def predict(self, X):
		# X here is a vector of distance
		# return 0 if X == 0 (exact equality under non-semantic embedding) otherwise 1
		return (1 - np.equal(X, 0)).astype(int)

	def predict_proba(self, X):
		is_equal = np.equal(X, 0).astype(float).reshape(-1,1)
		# X here is a vector of distance
		# return 0 if X == 0 (exact equality under non-semantic embedding) otherwise 1
		return np.hstack([is_equal, 1 - is_equal])


# use classifier first to classify items as 0 (paraphrases) or 1 (not paraphrases)
# then use these in the metrics as the distances

from copy import deepcopy
# SVC(class_weight='balanced')

# def classify_dmat(dmat, clf=LogisticRegression(class_weight='balanced')):
# 	# separate distances between paraphrases and not
# 	# diagnonals are paraphrases
# 	# make a deepcopy of the classifier to preserve the fit when used on different inputs
# 	clf = deepcopy(clf)
# 	paraphrase_dist = np.diag(dmat)
# 	# upper triangular, ignoring the diagonal; these are non-paraphrases
# 	nonparaphrase_dist = dmat[np.triu_indices(n=dmat.shape[0], k=1)]
# 	# 1 column of predictors
# 	X = np.concatenate((paraphrase_dist, nonparaphrase_dist)).reshape(-1,1)
# 	y = np.concatenate((np.zeros(len(paraphrase_dist)), np.ones(len(nonparaphrase_dist)))).astype(int)
# 	clf.fit(X, y)
#
# 	return clf

#  LogisticRegression(class_weight='balanced')
# SVC gets better separation
def classify_dmat(dmat, clf=LogisticRegression(class_weight='balanced')): #svm.SVC(probability=True, class_weight='balanced')):
	# dmat consists of a square matrix.  The upper left and lower right squares
	# are A and B vs themselves, respectively
	# the upper right is A vs B.  Extract the upper triangulars of each
	# in the first two, these are all 0 because they are d(a,a)=0 for each sentence a, so don't need to classify
	# since assume that equality of embeddings is equality of inputs

	# this function classifies paraphrase/same distances as 0 and 1 if others
	assert dmat.shape[0] == dmat.shape[1] # must be square matrix
	dim = dmat.shape[0]
	halfdim = int(dim/2)
	# make a deepcopy of the classifier to preserve the fit when used on different inputs
	clf = deepcopy(clf)
	A_vs_B = dmat[0:halfdim, halfdim:]
	paraphrase_dist = np.diag(A_vs_B)
	equal_dist = np.diag(dmat) # all zeros, or very close to
	# upper triangular indices for a square a quarter of the size of dmat
	upper_idx = np.triu_indices(n=halfdim, k=1)
	# take non-identical pairs within A and B, and non-paraphrases between A and B
	nonparaphrase_dist = np.concatenate((dmat[:halfdim, :halfdim][upper_idx], dmat[halfdim:, halfdim:][upper_idx], A_vs_B[upper_idx]))
	X = np.concatenate((paraphrase_dist, equal_dist, nonparaphrase_dist)).reshape(-1,1)
	# 0 if should be matched (i.e., have near 0 distance), 1 if not
	y = np.concatenate((np.zeros(len(paraphrase_dist) + len(equal_dist)), np.ones(len(nonparaphrase_dist)))).astype(int)
	clf.fit(X, y)

	# # # plot diagnostics
	# import matplotlib.pyplot as plt
	# import os
	# fig_dir = os.path.join(os.getcwd(), "../jupyter/figures")
	# num_par = len(paraphrase_dist)
	# num_equal = len(equal_dist)
	# ndists = X.shape[0]
	# dist_type = ['paraphrases'] * num_par + ['identical'] * num_equal + ['non-paraphrases'] * (ndists - num_par - num_equal)
	# Xdf = pd.DataFrame({'x': X[:, 0], 'y': y, 'type': dist_type})
	#
	# fig = plt.imshow(dmat, cmap='hot', vmin=0)
	# pwd = os.getcwd()
	# ax = plt.gca()
	# ax.set_title('A and B document cosine distances ' + r'$\delta$')
	# # make grid at half
	# ticks = np.arange(-0.5, dim + 0.5, halfdim)
	# tick_labels = [""] * len(ticks)
	# ax.set_xticks(ticks, labels=tick_labels, minor=False)
	# ax.set_yticks(ticks, labels=tick_labels, minor=False)
	# ax.grid(color='gray', linestyle='-', linewidth=0.5)
	# # labels at minor
	# label_locs = np.array([0.25, 0.75]) * dim
	# labels = ["A", "B"]
	# ax.set_xticks(label_locs, labels=labels, minor=True)
	# ax.set_yticks(label_locs, labels=labels, minor=True)
	#
	# plt.colorbar()
	# plt.savefig(os.path.join(fig_dir, 'cosine_distance_mat.png'))
	# plt.show()
	#
	# boundary = -1 * clf.intercept_[0] / clf.coef_[0]
	# for grp, pts in Xdf.groupby('type'):
	# 	plt.scatter(pts['x'], pts['y'], label=grp, alpha=0.5)
	# plt.xlabel('distances ' + r'$\delta$', fontsize='xx-small')
	# plt.ylabel('probability of non-match', fontsize='xx-small')
	# Xsort = np.sort(X, axis=0)
	# ypred = clf.predict_proba(Xsort)[:,1]
	# plt.plot(Xsort[:,0], ypred, color='red', marker='*', linestyle='--', label='classifier fit')
	# plt.xticks(fontsize='xx-small')
	# plt.yticks(fontsize='xx-small')
	#
	# plt.axhline(y=0.5, linestyle="--")
	# plt.axvline(x=boundary[0], linestyle="--")
	# plt.legend(fontsize='x-small')
	# plt.title('Semantic binarization of document distances ' + r'$\delta$', fontsize='small')
	# plt.savefig(os.path.join(fig_dir, 'semantic_binarization_cosine_distance_mat.pdf'))
	# plt.show()
	#
	# # # density plot
	# # add a small extra fake value of the zero distance
	# Xdf = pd.concat([Xdf, pd.DataFrame({'x': 0.0001, 'y': 0, 'type': 'identical'}, index=[99999])])
	# g = sns.kdeplot(data=Xdf, x='x', hue='type', common_norm=False, clip=[X.min(), X.max()], warn_singular=False)
	# plotted_lines = g.get_lines()
	# max_dens = max([plotted_lines[ii].get_data()[1].max() for ii in [0,2]]) * 1.025
	# # plt.legend(loc='upper center', fontsize='xx-small')
	#
	# lty = ['dotted', 'solid', 'dashed']
	# handles = g.legend_.legendHandles[::-1]
	# for line, lt, handle in zip(g.lines, lty, handles):
	# 	line.set_linestyle(lt)
	# 	handle.set_ls(lt)
	# g.set_ylim(top=max_dens)
	#
	# plt.xticks(fontsize='xx-small')
	# plt.yticks(fontsize='xx-small')
	# g.set_xlabel(xlabel='distance ' + r'$\delta$', fontsize='xx-small')
	# g.set_ylabel(ylabel='density', fontsize='xx-small')
	# plt.title('Document distances ' + r'$\delta$', fontsize='small')
	# plt.savefig(os.path.join(fig_dir, 'distribution_cosine_distance_mat.pdf'))
	# plt.show()
	#
	# dmat_binary = binarize_dmat_by_clf(arr=dmat, clf=clf, as_binary=True)
	# dmat_prob = binarize_dmat_by_clf(arr=dmat, clf=clf, as_binary=False)
	#
	# fig = plt.imshow(dmat_binary, cmap='Greys_r', vmin=0, vmax=1)
	# pwd = os.getcwd()
	# ax = plt.gca()
	# ax.set_title('A and B document binary distances ' + r'$\delta^*$')
	# # make grid at half
	# # ticks = np.arange(-0.5, dim + 0.5, halfdim)
	# ticks = np.linspace(-0.35, dim-1+0.1, num=3)
	#
	# tick_labels = [""] * len(ticks)
	# ax.set_xticks(ticks, labels=tick_labels, minor=False)
	# ax.set_yticks(ticks, labels=tick_labels, minor=False)
	# ax.grid(color='gray', linestyle='-', linewidth=0.5)
	# # labels at minor
	# label_locs = np.array([0.25, 0.75]) * dim
	# labels = ["A", "B"]
	# ax.set_xticks(label_locs, labels=labels, minor=True)
	# ax.set_yticks(label_locs, labels=labels, minor=True)
	#
	# plt.colorbar()
	# plt.savefig(os.path.join(fig_dir, 'binary_mapped_cosine_distance_mat.pdf'))
	# plt.show()
	#
	# fig = plt.imshow(dmat_prob, cmap='Greys_r', vmin=0, vmax=1)
	# pwd = os.getcwd()
	# ax = plt.gca()
	# ax.set_title('A and B document mapped distances ' + r'$\delta^*$')
	# # make grid at half
	# ticks = np.arange(-0.5, dim + 0.5, halfdim)
	# tick_labels = [""] * len(ticks)
	# ax.set_xticks(ticks, labels=tick_labels, minor=False)
	# ax.set_yticks(ticks, labels=tick_labels, minor=False)
	# ax.grid(color='gray', linestyle='-', linewidth=0.5)
	# # labels at minor
	# label_locs = np.array([0.25, 0.75]) * dim
	# labels = ["A", "B"]
	# ax.set_xticks(label_locs, labels=labels, minor=True)
	# ax.set_yticks(label_locs, labels=labels, minor=True)
	#
	# plt.colorbar()
	# plt.savefig(os.path.join(fig_dir, 'binary_prob_mapped_cosine_distance_mat.pdf'))
	# plt.show()

	return clf


# def compute_nearest_neighbour_distances_cosine(real_features, nearest_k):
# 	d = cosine_arccos_transform(c1=real_features) # self distance
# 	return pr.get_kth_value(d, k=nearest_k + 1, axis=-1)


# version with 0 and 1 binary distance
# def binarize_dmat_by_clf(arr, clf=None, as_binary=True):
# 	# take a continuous-valued array and binarize it
# 	if clf is not None:
# 		arr_flat = arr.flatten(order="C").reshape(-1, 1)
# 		if as_binary:
# 			arr_flat = clf.predict(X=arr_flat)
# 		else:
# 			arr_flat = clf.predict_proba(X=arr_flat)[:,1]
# 		return arr_flat.reshape(arr.shape)
# 	else:
# 		return arr

def binarize_dmat_by_clf(arr, clf=None, as_binary=True):
	# take a continuous-valued array and binarize it
	if clf is not None:
		arr_flat = arr.flatten(order="C").reshape(-1, 1)
		if as_binary:
			arr_flat_yhat = clf.predict(X=arr_flat).reshape(-1, 1)
		else:
			arr_flat_yhat = clf.predict_proba(X=arr_flat)[:,1]
		arr_flat[arr_flat != 0] = arr_flat_yhat[arr_flat != 0]
		return arr_flat.reshape(arr.shape)
	else:
		return arr


def compute_nearest_neighbour_distances_cosine(real_features, nearest_k, clf=None):
	d = cosine_arccos_transform(c1=real_features) # self distance
	# now binarize if passed a clf
	d = binarize_dmat_by_clf(arr=d, clf=clf)
	return pr.get_kth_value(d, k=nearest_k + 1, axis=-1)



def compute_prdc_cosine_clf(real_features, fake_features, nearest_k, clf=None):
	"""
    Computes precision, recall, density, and coverage given two manifolds.

    Args:
        real_features: numpy.ndarray([N, feature_dim], dtype=np.float32)
        fake_features: numpy.ndarray([N, feature_dim], dtype=np.float32)
        nearest_k: int.
    Returns:
        dict of precision, recall, density, and coverage.
    """

	print('Num real: {} Num fake: {}'
          .format(real_features.shape[0], fake_features.shape[0]))

	real_nearest_neighbour_distances = compute_nearest_neighbour_distances_cosine(
        real_features, nearest_k, clf)
	fake_nearest_neighbour_distances = compute_nearest_neighbour_distances_cosine(
        fake_features, nearest_k, clf)
	distance_real_fake = cosine_arccos_transform(c1=real_features, c2=fake_features)
	distance_real_fake = binarize_dmat_by_clf(arr=distance_real_fake, clf=clf)
	# # now binarize
	# if clf is not None:
	# 	# convert to binary using clf
	# 	real_nearest_neighbour_distances_flat = clf.predict(X=real_nearest_neighbour_distances.reshape(-1,1))
	# 	# real_nearest_neighbour_distances_flat[real_nearest_neighbour_distances == 0] = 0
	# 	real_nearest_neighbour_distances = real_nearest_neighbour_distances_flat
	#
	# 	fake_nearest_neighbour_distances_flat = clf.predict(X=fake_nearest_neighbour_distances.reshape(-1,1))
	# 	# fake_nearest_neighbour_distances_flat[fake_nearest_neighbour_distances == 0] = 0
	# 	fake_nearest_neighbour_distances = fake_nearest_neighbour_distances_flat
	#
	# 	distance_real_fake_flat = distance_real_fake.flatten(order="C").reshape(-1,1)
	# 	distance_real_fake_flat = clf.predict(X=distance_real_fake_flat)
	# 	# distance_real_fake_flat[distance_real_fake_flat == 0] = 0
	# 	distance_real_fake = distance_real_fake_flat.reshape(distance_real_fake.shape)

	precision = (
            distance_real_fake <
            np.expand_dims(real_nearest_neighbour_distances, axis=1)
    ).any(axis=0).mean()

	recall = (
            distance_real_fake <
            np.expand_dims(fake_nearest_neighbour_distances, axis=0)
    ).any(axis=1).mean()

	density = (1. / float(nearest_k)) * (
            distance_real_fake <
            np.expand_dims(real_nearest_neighbour_distances, axis=1)
    ).sum(axis=0).mean()

	coverage = (
            distance_real_fake.min(axis=1) <
            real_nearest_neighbour_distances
    ).mean()

	return dict(precision=precision, recall=recall,
                density=density, coverage=coverage)




def classifier_distance_clf(corpus1: Corpus, corpus2: Corpus, clf=None, model: TextEmbedder = STTokenizerEmbedder()):
	# distance between corpora is the F1 score of a classifier trained to classify membership of a random sample of each
	embeddings1, embeddings2 = utils.get_corpora_embeddings(corpus1, corpus2, model)

	corpus1_vecs = embeddings1
	corpus1_train_indx = random.choice(a=len(embeddings1), size=int(0.8 * len(embeddings1)))
	corpus1_train = itemgetter(*corpus1_train_indx)(corpus1_vecs)

	corpus1_test_indx = set(range(len(embeddings1))) - (set(corpus1_train_indx))
	corpus1_test = itemgetter(*corpus1_test_indx)(corpus1_vecs)

	corpus2_vecs = embeddings2
	corpus2_train_indx = random.choice(a=len(embeddings2), size=int(0.8 * len(embeddings2)))
	corpus2_train = itemgetter(*corpus2_train_indx)(corpus2_vecs)

	corpus2_test_indx = set(range(len(embeddings2))) - (set(corpus2_train_indx))
	corpus2_test = itemgetter(*corpus2_test_indx)(corpus2_vecs)

	train_x = corpus1_train + corpus2_train
	train_y = [0] * len(corpus1_train) + [1] * len(corpus2_train)
	test_x = corpus1_test + corpus2_test
	test_y = [0] * len(corpus1_test) + [1] * len(corpus2_test)

	# calculate Euclidean distances
	# from sklearn.metrics.pairwise import euclidean_distances
	# pairwise distances, rather than observations themselves
	dmat_tr = cosine_arccos_transform(c1=train_x)
	dmat_tr = binarize_dmat_by_clf(arr=dmat_tr, clf=clf)

	# fit a model on the precomputed distance matrix
	model = svm.SVC(random_state=1, kernel='precomputed')
	model.fit(dmat_tr, train_y)

	# test set distance matrix between test and train objects
	dmat_te = cosine_arccos_transform(c1=test_x, c2=train_x)
	dmat_te = binarize_dmat_by_clf(arr=dmat_te, clf=clf)

	y_pred = model.predict(dmat_te)
	correct = f1_score(test_y, y_pred)

	return correct

# def medoid_clf(corpus1, corpus2, model: TextEmbedder = STTokenizerEmbedder()):
#
# 	embeddings1, embeddings2 = utils.get_corpora_embeddings(corpus1, corpus2, model)
# 	# from scipy import spatial
# 	# # calculate mean and covariance statistics
# 	# act1 = np.vstack(embeddings1)
# 	# act2 = np.vstack(embeddings2)
# 	# mu1 = np.mean(act1, axis=0)
# 	# mu2 = np.mean(act2, axis=0)
# 	# # calculate sum squared difference between means
# 	# cosine = spatial.distance.cosine(mu1, mu2)
# 	# #
# 	# #
# 	# #
# 	# # cosine = np.clip(cosine_similarity(embeddings1, embeddings2), -1, 1)
# 	# # table = np.arccos(cosine) / np.pi
#
# 	cosine = np.clip(cosine_similarity(embeddings1, embeddings2), -1, 1)
#
#
# 	return classify_dmat(dmat=cosine)



def cosine_clf(corpus1, corpus2, model: TextEmbedder = STTokenizerEmbedder()):
	# receive two whole corpora; calculate cosine distance between all pairs of documents
	# classify_dmat builds a classifer that separates pairwise distances between paraphrases/same documents to all others
	embeddings1, embeddings2 = utils.get_corpora_embeddings(corpus1, corpus2, model)
	# need to calculate distances between all pairs, not just between the corpora

	table = cosine_arccos_transform(c1=np.concatenate((embeddings1, embeddings2)))

	return classify_dmat(dmat=table)

def IRPR_distance_clf(corpus1: Corpus, corpus2: Corpus, clf=None, model: TextEmbedder = STTokenizerEmbedder()):
	embeddings1, embeddings2 = utils.get_corpora_embeddings(corpus1, corpus2, model)

	# cosine = np.clip(cosine_similarity(embeddings1, embeddings2), -1, 1)
	# # this is because sometimes cosine_similarity return values larger than 1
	# table = np.arccos(cosine) / np.pi
	# is_zero_dist = (table == 0)
	table = cosine_arccos_transform(embeddings1, embeddings2)
	table = binarize_dmat_by_clf(arr=table, clf=clf)

	precision = np.nansum(np.nanmin(table, axis=1)) / table.shape[1]
	recall = np.nansum(np.nanmin(table, axis=0)) / table.shape[0]
	return 2 * (precision * recall) / (precision + recall) #(precision + recall)/2 #


def Directed_Hausdorff_distance_clf(corpus1: Corpus, corpus2: Corpus, model: TextEmbedder = STTokenizerEmbedder(), clf=None):
	# calculate nearest distance from each element in one corpus to an element in the other
	# like IRPR except take mean not harmonic mean (F1-score)
	if model is not None:
		embeddings1, embeddings2 = utils.get_corpora_embeddings(corpus1, corpus2, model)
	else:
		embeddings1, embeddings2 = corpus1, corpus2

	table = cosine_arccos_transform(c1=embeddings1, c2=embeddings2)
	table = binarize_dmat_by_clf(arr=table, clf=clf)

	nearest_1to2 = np.nanmin(table, axis=1) # nearest in c2 from each in c1, min in each row
	nearest_2to1 = np.nanmin(table, axis=0)  # nearest in c1 from each in c2, min in each column

	return np.mean([nearest_1to2.mean(), nearest_2to1.mean()])


def Energy_distance_clf(corpus1: Corpus, corpus2: Corpus, model: TextEmbedder = STTokenizerEmbedder(), normalize=False, clf=None):
	# https://en.wikipedia.org/wiki/Energy_distance
	if model is not None:
		embeddings1, embeddings2 = utils.get_corpora_embeddings(corpus1, corpus2, model)
	else:
		embeddings1, embeddings2 = corpus1, corpus2

	between = binarize_dmat_by_clf(arr=cosine_arccos_transform(c1=embeddings1, c2=embeddings2), clf=clf)
	within1 = binarize_dmat_by_clf(arr=cosine_arccos_transform(c1=embeddings1), clf=clf)
	within2 = binarize_dmat_by_clf(arr=cosine_arccos_transform(c1=embeddings2), clf=clf)

	A2 = 2 * between.mean()
	B = within1.mean()
	C = within2.mean()

	edist = A2 - B - C
	#  E-coefficient of inhomogeneity is between 0 and 1
	return edist/A2 if normalize else np.sqrt(edist)



def Euclidean_clf(corpus1: Corpus, corpus2: Corpus, model: TextEmbedder = STTokenizerEmbedder()):
	embeddings1, embeddings2 = utils.get_corpora_embeddings(corpus1, corpus2, model)
	# just pairwise Euclidean distance
	# d = pr.compute_pairwise_distance(embeddings1, embeddings2)
	d = pr.compute_pairwise_distance(data_x=np.vstack((embeddings1, embeddings2)))
	d[ d <= ZERO_THRESH] = 0.0

	return classify_dmat(dmat=d)

def compute_prdc_clf(real_features, fake_features, clf, nearest_k=5):

	real_nearest_neighbour_distances = pr.compute_nearest_neighbour_distances(
		real_features, nearest_k)
	fake_nearest_neighbour_distances = pr.compute_nearest_neighbour_distances(
		fake_features, nearest_k)
	distance_real_fake = pr.compute_pairwise_distance(
		real_features, fake_features)

	# convert very small values to 0 exactly
	real_nearest_neighbour_distances[ real_nearest_neighbour_distances <= ZERO_THRESH] = 0.0
	fake_nearest_neighbour_distances[fake_nearest_neighbour_distances <= ZERO_THRESH] = 0.0
	distance_real_fake[distance_real_fake <= ZERO_THRESH] = 0.0

	# now binarize
	real_nearest_neighbour_distances_flat = clf.predict(X=real_nearest_neighbour_distances.reshape(-1,1))
	# real_nearest_neighbour_distances_flat[real_nearest_neighbour_distances == 0] = 0
	real_nearest_neighbour_distances = real_nearest_neighbour_distances_flat

	fake_nearest_neighbour_distances_flat = clf.predict(X=fake_nearest_neighbour_distances.reshape(-1,1))
	# fake_nearest_neighbour_distances_flat[fake_nearest_neighbour_distances == 0] = 0
	fake_nearest_neighbour_distances = fake_nearest_neighbour_distances_flat

	distance_real_fake_flat = distance_real_fake.flatten(order="C").reshape(-1,1)
	distance_real_fake_flat = clf.predict(X=distance_real_fake_flat)
	# distance_real_fake_flat[distance_real_fake_flat == 0] = 0
	distance_real_fake = distance_real_fake_flat.reshape(distance_real_fake.shape)

	# now calculate measures on the binarized (or transformed) data

	precision = (
			distance_real_fake <
			np.expand_dims(real_nearest_neighbour_distances, axis=1)
	).any(axis=0).mean()

	recall = (
			distance_real_fake <
			np.expand_dims(fake_nearest_neighbour_distances, axis=0)
	).any(axis=1).mean()

	density = (1. / float(nearest_k)) * (
			distance_real_fake <
			np.expand_dims(real_nearest_neighbour_distances, axis=1)
	).sum(axis=0).mean()

	coverage = (
			distance_real_fake.min(axis=1) <
			real_nearest_neighbour_distances
	).mean()

	return dict(precision=precision, recall=recall,
				density=density, coverage=coverage)


def pr_distance_clf(corpus1: Corpus, corpus2: Corpus,  clf, model: TextEmbedder = STTokenizerEmbedder(), nearest_k=5, cosine=False):
	embeddings1, embeddings2 = utils.get_corpora_embeddings(corpus1, corpus2, model)

	f = compute_prdc_cosine_clf if cosine else compute_prdc_clf

	metric = f(real_features=np.vstack(embeddings1),
			   fake_features=np.vstack(embeddings2),
			   clf=clf,
			   nearest_k=nearest_k)
	precision = np.clip(metric['precision'], 0, 1)
	recall = np.clip(metric['recall'] + 1e-6, 0, 1)

	return  1 - 2 * (precision * recall) / (precision + recall)# 1 - (precision + recall)/2


def dc_distance_clf(corpus1: Corpus, corpus2: Corpus,  clf, model: TextEmbedder = STTokenizerEmbedder(), nearest_k=5, cosine=False):
	embeddings1, embeddings2 = utils.get_corpora_embeddings(corpus1, corpus2, model)

	f = compute_prdc_cosine_clf if cosine else compute_prdc_clf

	metric = f(real_features=np.vstack(embeddings1),
			   fake_features=np.vstack(embeddings2),
			   clf=clf,
			   nearest_k=nearest_k)

	density = np.clip(metric['density'], 0, 1)
	coverage = np.clip(metric['coverage'] + 1e-6, 0, 1)

	return 1 - 2 * (density * coverage) / (density + coverage) # 1 - (density + coverage) / 2#

