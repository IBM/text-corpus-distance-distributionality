import time
import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.linear_model
from compcor.utils import Corpus, TCorpus
from meme.KSC import KSC
from compcor.corpus_metrics import  *
from meme.metric_characteristics import *
# from experiments.meme.experiment_config import get_metric_dependant_data, output_format, ksc_measures
from meme.experiment_config import ksc_measures
from utils import sort_numsuffix
from collections import namedtuple

output_format = '{}_{}_{}_'
distance_dtypes = {'repetition': int, 'metric': str, 'distance_score': float, 'i': int, 'j': int, 'l': int}

def robust_standardization(x):
	med = np.median(x)
	mad = np.abs(x - med).mean()
	return (x - med) / mad


# overwrite get_metric_dependant data from meme, here only do tokenizer if not already in float form
def get_metric_dependant_data(metric, corpus: Corpus):
	if isinstance(corpus[0], str):
		# if is text corpus, not already embedded
		if metric in (zipf_distance, chi_square_distance):
			c = STTokenizerEmbedder().tokenize_sentences(corpus)
		else:
			c = STTokenizerEmbedder().embed_sentences(corpus)
		return c
	else:
		# is already embedded
		return corpus


SMALL_SIZE = 15
matplotlib.rc('font', size=SMALL_SIZE)
matplotlib.rc('axes', titlesize=SMALL_SIZE)

sns.set_theme(style="whitegrid")
sns.set(font_scale=2)
from copy import deepcopy

# def runKSC(metrics, metric_names, corpus1:Corpus, corpus2:Corpus, n=30, k=7, repetitions=5, output="ksc_results/data", coverage=False):
# 	ksc_results = []
# 	distance_results = []
#
# 	for metric_idx, metric in enumerate(metrics):
# 		print('Distance by {} *******************'.format(metric))
# 		c1 = get_metric_dependant_data(metric, corpus1)
# 		c2 = get_metric_dependant_data(metric, corpus2)
#
# 		print(c1)
# 		print(c2)
#
# 		for rep in range(repetitions):
#
# 			distances_metric = []
#
# 			ksc = KSC._known_similarity_corpora(deepcopy(c1), deepcopy(c2), n=n, k=k, unique_samples_corpora=not coverage)
# 			start = time.time()
# 			# ksc_score, ksc_score_weighted, distance_stats = KSC.test_ksc(ksc, dist=metric, plot_result=False)
# 			ksc_score, ksc_score_weighted, distance_stats = KSC.test_ksc(ksc, dist=metric)
# 			print('done test ksc')
# 			ksc_time = (time.time() - start) / len(distance_stats)
#
# 			# if coverage:
# 			# 	polindrome = [i*(k-i)/(k**2) for i in range(k)]
# 			# 	distances_metric.append(
# 			# 		np.vstack([[metric_names[metric_idx], rep, a, b, polindrome[b] + polindrome[a], y] for (a, b, y) in distance_stats]))
# 			# else:
# 			# 	distances_metric.append(
# 			# 		np.vstack([[metric_names[metric_idx], rep, a, b, b - a, y] for (a, b, y) in distance_stats]))
# 			distances_metric.append(
# 				np.vstack([[metric_names[metric_idx], rep, a, b, b - a, y] for (a, b, y) in distance_stats]))
#
# 			distances_metric = np.vstack(distances_metric)
# 			# return distances_metric, ksc_time
#
# 			# normalize the score for a specific metric.
# 			distances_metric = np.append(distances_metric, sklearn.preprocessing.StandardScaler().fit_transform(
# 				distances_metric[:, 5].reshape(-1, 1)), axis=1)
# 			distance_results.extend(distances_metric)
#
# 			ells = distances_metric[:, 4].astype('float')
# 			ds_normalized = distances_metric[:, 6].astype('float')
#
# 			monotonicity = metric_monotonicity(ells, ds_normalized)
# 			separability = metric_separability(ells, ds_normalized)
# 			linearity = metric_linearity(ells, ds_normalized)
# 			ksc_results.append(
# 				[metric_names[metric_idx], ksc_score, ksc_score_weighted, ksc_time, monotonicity, separability, linearity])
#
# 	metrics_measures_df = pd.DataFrame(data=ksc_results, columns=['metric'] + ksc_measures)
# 	metrics_measures_df['Time'] = (1 / metrics_measures_df['Time'])/100
#
# 	all_distance_samples_df = pd.DataFrame(data=distance_results,
# 									columns=['metric', 'repetition', 'i', 'j', 'l', 'distance', 'distance_score'])
# 	all_distance_samples_df["l"] = pd.to_numeric(all_distance_samples_df["l"])
# 	all_distance_samples_df["distance"] = all_distance_samples_df["distance"].astype(float)
# 	all_distance_samples_df["distance_score"] = all_distance_samples_df["distance_score"].astype(float)
# 	#output_pattern = os.path.join(output,output_format.format(corpus1.name,corpus2.name, n,k))
# 	output_pattern = os.path.join(output, output_format.format('quora', n,k))
#
# 	metrics_measures_df.to_csv(path_or_buf=output_pattern+'metrics_measures_df.csv', index=False,
# 					  float_format='%.3f')
# 	all_distance_samples_df.to_csv(path_or_buf=output_pattern+'all_distance_samples_df.csv', index=False,
# 							   float_format='%.3f')
#
# 	return metrics_measures_df, all_distance_samples_df

def calcKSC_scores_fixed_sample(c1, c2, indices_from_each, metric, metric_name, rep=0):
	# the actual ksc corpora themselves
	ksc_corpora = [np.vstack([c1[idx[0], :], c2[idx[1], :]]) for idx in indices_from_each]

	start = time.time()
	accuracy, weighted_accuracy, distance_stats = KSC.test_ksc(ksc_corpora, dist=metric)
	ksc_time = (time.time() - start) / len(distance_stats)

	# if coverage:
	# 	polindrome = [i*(k-i)/(k**2) for i in range(k)]
	# 	distances_metric.append(
	# 					np.vstack([[metric_names[metric_idx], rep, a, b, polindrome[b] + polindrome[a], y] for (a, b, y) in distance_stats]))
	# else:
	tmp = [[metric_name, rep, a, b, b - a, y] for (a, b, y) in distance_stats]
	if isinstance(tmp[0][-1], tuple):
		# distance returned as namedtuple of components
		distances_metric = pd.DataFrame(data=np.vstack([row[:-1] for row in tmp]),
										columns=['metric', 'repetition', 'i', 'j', 'l'])
		distances_metric['distance'] = [row[-1] for row in tmp]
		orig_d = np.array([row.distance for row in distances_metric['distance']]).reshape(-1,1)
		distances_metric['robust_std_distance'] = robust_standardization(orig_d)
		distances_metric['distance_score'] = sklearn.preprocessing.StandardScaler().fit_transform(
			orig_d)

	else:
		distances_metric = pd.DataFrame(data=np.vstack(tmp),
										columns=['metric', 'repetition', 'i', 'j', 'l', 'distance'])
		distances_metric['distance'] = distances_metric['distance'].astype(float)
		distances_metric['robust_std_distance'] = robust_standardization(distances_metric['distance'])
		distances_metric['distance_score'] = sklearn.preprocessing.StandardScaler().fit_transform(
			distances_metric[['distance']])

	distances_metric = distances_metric.astype(dtype=distance_dtypes)
	ells = distances_metric['l'].astype('float')

	monotonicity = metric_monotonicity(ells, distances_metric['distance_score'])
	separability = metric_separability(ells, distances_metric['distance_score'])
	linearity = metric_linearity(ells, distances_metric['distance_score'])
	# ksc_results.append(
		# [metric_names[metric_idx], accuracy, weighted_accuracy, ksc_time, monotonicity, separability, linearity])
	ksc_res = [metric_name, accuracy, weighted_accuracy, ksc_time, monotonicity, separability, linearity]
	return distances_metric, ksc_res



def runKSC_fixed_sample(metrics, metric_names, corpus1:Corpus, corpus2:Corpus, n=30, k=7, repetitions=5, output="ksc_results/data", coverage=False):
	# run KSC but keep the same indices sampled and return them
	ksc_results = []
	distance_results = []

	lenc1, lenc2 = len(corpus1), len(corpus2)
	min_n = min(lenc1, lenc2)
	if n < min_n:
		raise ValueError("n must be at least as large as the smaller corpus size of {}".format(min_n))
	if n < k:
		print('Note: since n < k the proportions sampled corpus1 may be the same in two or more successive KSC corpora')

	c1idx = np.arange(lenc1).reshape(-1,1)
	c2idx = np.arange(start=lenc1, stop=lenc1 + lenc2).reshape(-1,1)
	# transform each corpus into the form required by the metric
	c1 = [get_metric_dependant_data(metric, corpus1) for metric in metrics]
	c2 = [get_metric_dependant_data(metric, corpus2) for metric in metrics]
	# keep for return, keep the same separation for all metrics
	ksc_corpora_indices = []

	for rep in range(repetitions):
		# collect the indices, reverse order of c1 and c2, so that index 0 is all from c1 (0 from c2)
		ksc_corpora_indices.append(KSC._known_similarity_corpora(c2idx, c1idx, n=n, k=k, unique_samples_corpora=not coverage))
		# indices of the corpora selected
		ksc_corpora_indices[-1] = [np.array(idx).reshape(1,-1)[0] for idx in ksc_corpora_indices[-1]]
		are_from_c1 = [idx < lenc1 for idx in ksc_corpora_indices[-1]]
		# now select the indices separately from each corpus, keeping them fixed across metrics
		ksc_c1_idxs = [idx[memb] for idx, memb in zip(ksc_corpora_indices[rep], are_from_c1)]
		ksc_c2_idxs = [idx[np.logical_not(memb)] - lenc1 for idx, memb in zip(ksc_corpora_indices[-1], are_from_c1)]
		# replace them
		ksc_corpora_indices[-1] = [[idx1, idx2] for idx1, idx2 in zip(ksc_c1_idxs, ksc_c2_idxs)]

		for metric_idx, metric in enumerate(metrics):
			d_res, ksc_res = calcKSC_scores_fixed_sample(c1=c1[metric_idx], c2=c2[metric_idx], indices_from_each=ksc_corpora_indices[-1],
														 metric=metric, metric_name=metric_names[metric_idx])
			distance_results.append(d_res)
			ksc_results.append(ksc_res)


	metrics_measures_df = pd.DataFrame(data=ksc_results, columns=['metric'] + ksc_measures)
	metrics_measures_df['Time'] = (1 / metrics_measures_df['Time'])/100

	# combine all results
	all_distance_samples_df = pd.concat(distance_results, ignore_index=True)
	all_distance_samples_df.reset_index(drop=True, inplace=True)

	# all_distance_samples_df["l"] = pd.to_numeric(all_distance_samples_df["l"])
	# use astype(float) because can handle NaN as strings
	# all_distance_samples_df["distance"] = all_distance_samples_df["distance"].astype(float)
	# all_distance_samples_df["distance_score"] = all_distance_samples_df["distance_score"].astype(float)
	# all_distance_samples_df["distance"] = pd.to_numeric(all_distance_samples_df["distance"])
	# all_distance_samples_df["distance_score"] = pd.to_numeric(all_distance_samples_df["distance_score"])
	if output is not None:
		output_pattern = os.path.join(output, output_format.format('quora', n, k))
		metrics_measures_df.to_csv(path_or_buf=output_pattern+'metrics_measures_df.csv', index=False, float_format='%.3f')
		all_distance_samples_df.to_csv(path_or_buf=output_pattern+'all_distance_samples_df.csv', index=False, float_format='%.3f')

	return metrics_measures_df, all_distance_samples_df, ksc_corpora_indices




def runKSC(metrics, metric_names, corpus1:Corpus, corpus2:Corpus, n=30, k=7, repetitions=5, output="ksc_results/data", coverage=False):
	ksc_results = []
	distance_results = []

	for metric_idx, metric in enumerate(metrics):
		c1 = get_metric_dependant_data(metric, corpus1)
		c2 = get_metric_dependant_data(metric, corpus2)

		for rep in range(repetitions):

			distances_metric = []

			ksc = KSC._known_similarity_corpora(c1, c2, n=n, k=k, unique_samples_corpora=not coverage)
			start = time.time()
			accuracy, weighted_accuracy, distance_stats = KSC.test_ksc(ksc, dist=metric)
			ksc_time = (time.time() - start) / len(distance_stats)

			# if coverage:
			# 	polindrome = [i*(k-i)/(k**2) for i in range(k)]
			# 	distances_metric.append(
			# 					np.vstack([[metric_names[metric_idx], rep, a, b, polindrome[b] + polindrome[a], y] for (a, b, y) in distance_stats]))
			# else:

			tmp = [[metric_names[metric_idx], rep, a, b, b - a, y] for (a, b, y) in distance_stats]
			if isinstance(tmp[0][-1], tuple):
				# distance returned as namedtuple of components
				distances_metric = pd.DataFrame(data=np.vstack([row[:-1] for row in tmp]),
												columns=['metric', 'repetition', 'i', 'j', 'l'])
				distances_metric['distance'] = [row[-1] for row in tmp]
				distances_metric['distance_score'] = sklearn.preprocessing.StandardScaler().fit_transform(
					np.array([row.distance for row in distances_metric['distance']]).reshape(-1, 1))

			else:
				distances_metric = pd.DataFrame(data=np.vstack(tmp),
												columns=['metric', 'repetition', 'i', 'j', 'l', 'distance'])
				distances_metric['distance'] = distances_metric['distance'].astype(float)
				distances_metric['distance_score'] = sklearn.preprocessing.StandardScaler().fit_transform(
					distances_metric[['distance']])
			# # normalize the score for a specific metric.
			# tmp = distances_metric[:, 5].astype(float)
			# med = np.median(tmp)
			# # mean absolute deviation
			# mad = np.abs(tmp - med).mean()
			# distances_metric = np.append(distances_metric, (tmp.reshape(-1,1) - med) / mad, axis=1)
			# distance_results.extend(distances_metric)

			distances_metric = distances_metric.astype(dtype=distance_dtypes)

			# # normalize the score for a specific metric.
			# tmp = distances_metric[:, 5].astype(float)
			# med = np.median(tmp)
			# # mean absolute deviation
			# mad = np.abs(tmp - med).mean()
			# distances_metric = np.append(distances_metric, (tmp.reshape(-1, 1) - med) / mad, axis=1)

			distance_results.append(distances_metric)

			ells = distances_metric['l'].astype('float')

			monotonicity = metric_monotonicity(ells, distances_metric['distance_score'])
			separability = metric_separability(ells, distances_metric['distance_score'])
			linearity = metric_linearity(ells, distances_metric['distance_score'])
			ksc_results.append(
				[metric_names[metric_idx], accuracy, weighted_accuracy, ksc_time, monotonicity, separability, linearity])

	metrics_measures_df = pd.DataFrame(data=ksc_results, columns=['metric'] + ksc_measures)
	metrics_measures_df['Time'] = (1 / metrics_measures_df['Time'])/100

	all_distance_samples_df = pd.concat(distance_results, ignore_index=True)


	# all_distance_samples_df["l"] = pd.to_numeric(all_distance_samples_df["l"])
	# # use astype(float) because can handle NaN as strings
	# all_distance_samples_df["distance"] = all_distance_samples_df["distance"].astype(float)
	# all_distance_samples_df["distance_score"] = all_distance_samples_df["distance_score"].astype(float)
	# all_distance_samples_df["distance"] = pd.to_numeric(all_distance_samples_df["distance"])
	# all_distance_samples_df["distance_score"] = pd.to_numeric(all_distance_samples_df["distance_score"])
	if output is not None:
		output_pattern = os.path.join(output, output_format.format('quora', n, k))
		metrics_measures_df.to_csv(path_or_buf=output_pattern+'metrics_measures_df.csv', index=False, float_format='%.3f')
		all_distance_samples_df.to_csv(path_or_buf=output_pattern+'all_distance_samples_df.csv', index=False, float_format='%.3f')

	return metrics_measures_df, all_distance_samples_df

def plotKSC(all_distance_samples_df, boxplot=True, standardized=False, ncolumns=6, suffix='', fname=None, output=None, add_line=True):

	sns.set_theme(style="whitegrid")
	sns.set(font_scale=1.3)

	distance_feat = 'distance_score' if standardized else 'distance'
	# distance_feat = 'robust_std_distance' if standardized else 'distance'
	# if boxplot:
	# 	all_distance_samples_df['l'] = all_distance_samples_df['l'].astype(str)

	metrics_names = np.unique(all_distance_samples_df['metric']).tolist()
	metrics_names = sorted(metrics_names, key=sort_numsuffix)

	nmetrics = len(metrics_names)
	ncolumns = min(ncolumns, nmetrics)
	nrows = int(np.ceil(nmetrics / ncolumns))
	# number of empty cells
	nempty = (nrows * ncolumns) - nmetrics

	fig, axs = plt.subplots(nrows, ncolumns, figsize=(6 * ncolumns, 5 * nrows))
	for ax, metric in zip(fig.axes, metrics_names):
		metric_df = all_distance_samples_df.loc[all_distance_samples_df['metric'] == metric]
		# sns.swarmplot(x='l', y='distance_score', data= metric_df, color=".25",ax=axlist[i])
		# sns.boxplot(x='l', y='distance_score', data=metric_df, ax=axlist[i], color='b')
		# metric_df['l']=metric_df['l']-1 #woraround a defect in sns

		if boxplot:
			sns.boxplot(x='l', y=distance_feat, data=metric_df, ax=ax, color='orange')
			if add_line:
				sns.regplot(x='l', y=distance_feat, data=metric_df, ax=ax, scatter=False)

		else:
			sns.scatterplot(x='l', y=distance_feat, data=metric_df, ax=ax, color='orange')
			sns.regplot(x='l', y=distance_feat, data=metric_df, ax=ax,
						scatter=False, truncate=False)
		ax.set_title('{}'.format(metric), fontsize='x-large')
		ax.set_xlabel(r'$\ell$ (KSC separation)')
		if not standardized:
			# make sure to include 0 at bottom
			y = ax.get_ylim()
			d = y[1] - y[0]
			ax.set_ylim(bottom=-0.03 * d)
	# [axi.set(xlabel=None) for axi in axs]
	# [axi.set(ylabel=None) for axi in axs]

	for idx in range(1, nempty+1):
		# get rid of empty frames at the end
		axs.flat[-idx].set_visible(False)
	#[axi.set(ylim=(np.min(all_distance_samples_df['distance']),
	#			   np.max(all_distance_samples_df['distance']))) for axi in axlist]

	# plt.subplots_adjust(left=0.05,
	# 					bottom=0.1,
	# 					right=0.99,
	# 					top=0.9,
	# 					wspace=0.3,
	# 					hspace=0.4)
	plt.tight_layout()

	if output is not None:
		if fname is None:
			# use automatic fname
			fname = '{}_{}{}.png'.format('boxplot' if boxplot else 'scatter', 'standard' if standardized else 'raw', suffix)
		plt.savefig(os.path.join(str(output), fname))

	plt.show()





