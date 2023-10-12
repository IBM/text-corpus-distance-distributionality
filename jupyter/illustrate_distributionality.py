from compcor.corpus_metrics import  *
import numpy as np
import pandas as pd
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
from copy import deepcopy
import seaborn as sns
from inspect import signature
from sklearn.neighbors import NearestNeighbors
from utils import sort_numsuffix

def ppoints_to_n(p, X):
	ppoints = np.clip(p, a_min=0.0, a_max=1.0)
	return np.round(ppoints * X.shape[0]).astype(int)

	
class BlobSample:
	def __init__(self, n_samples=100, n_features=2, cluster_std=1.0):
		self.n_samples = n_samples
		self.cluster_std = cluster_std
		self.X, self.cluster_assignment, self.cluster_centers = make_blobs(centers=None, n_samples=self.n_samples, n_features=n_features, cluster_std=self.cluster_std,
																		   return_centers=True)
		self.n_features = self.X.shape[1]

	def gen_other(self, pair_proportion=1.0, jitter_std=0.1):
		# generate a pairing
		assert (pair_proportion >= 0) and (pair_proportion <= 1)
		assert (jitter_std >= 0)

		# some pairing
		n_paired = int(np.round(self.n_samples * pair_proportion))
		n_not_paired = self.n_samples - n_paired
		X = np.zeros((self.n_samples, self.n_features))

		paired_indices = np.random.choice(a=self.n_samples, size=n_paired, replace=False)
		not_paired_indices = np.array([ii for ii in range(self.n_samples) if ii not in paired_indices])

		if n_paired > 0:
			jitter_cmat = np.diag([jitter_std] * self.n_features)
			# jitter selected rows by random noise
			X[paired_indices,:] = np.vstack([np.random.multivariate_normal(mean=x, cov=jitter_cmat) for x in self.X[paired_indices,:]])
			# X[paired_indices,:] = self.X[paired_indices,:] + random_direction_noise(npoints=n_paired, ndim=self.n_features, magnitude=jitter_std)

		#np.vstack([np.random.multivariate_normal(mean=x, cov=jitter_cmat) for x in self.X[paired_indices,:]])

		if n_not_paired > 0:
			cluster_cmat = np.diag([self.cluster_std] * self.n_features)
			# do like make_blobs
			# use original assignments of the selected points, but don't directly pair the points
			# need this so that when we do perturb_group, we can choose items from the same group
			# assignments = np.random.choice(a=self.cluster_assignment, size=n_not_paired, replace=False)
			assignments = self.cluster_assignment[not_paired_indices]
			X[not_paired_indices,:] = np.vstack([np.random.multivariate_normal(mean=x, cov=cluster_cmat) for x in self.cluster_centers[assignments,:]])

		return X, paired_indices



	def perturb_grouped_points(self, X1=None, ppoints=0.1, perturb_radius=0.1, force_away=False, draw=False):
		from collections import Counter
		# perturb points from X1 (or X0.X if X1 is None) in a group in the same direction
		# force_away: force it to be away from the cluster centroid
		if X1 is not None:
			assert X1.shape == self.X.shape
			X_p = deepcopy(X1)
		else:
			X_p = deepcopy(self.X)
		X = deepcopy(X_p)

		npoints = ppoints_to_n(p=ppoints, X=X_p)
		n_features = self.n_features
		perturbed_indices = []
		if npoints > 0:
			# find nearest neighbors of a random point
			nbrs = NearestNeighbors(n_neighbors=npoints, algorithm='ball_tree').fit(X)
			# note indices includes the point itself
			perturbed_indices = nbrs.kneighbors(X[np.random.choice(a=len(X), size=1), :], return_distance=False)[0,:]
			if force_away:
				# shift in direction away from the center of the selected source point
				vec_from_centroid = self.cluster_centers[self.cluster_assignment[perturbed_indices[0]],:]
				# rescale so has given magnitude
				shift = X[perturbed_indices[0],:] - vec_from_centroid
				shift = perturb_radius * shift / np.linalg.norm(shift, axis=0)
			else:
				# random direction shift
				shift = random_direction_noise(npoints=1, ndim=n_features, magnitude=perturb_radius)
			X_p[perturbed_indices, :] = np.vstack([xx + shift for xx in X_p[perturbed_indices, :]])

			# choose clusters to start with, in random order
			# choose centroids to move together
			# nclusters = self.cluster_centers.shape[0]

			# centroid_ids = np.random.choice(a=nclusters, size=nclusters, replace=False)
			# ctr = Counter(self.cluster_assignment)
			# perturbed_indices = []
			# if npoints > 0:
			# 	for cl in centroid_ids:
			# 		ntodraw = min(ctr[cl], npoints - len(perturbed_indices))
			# 		belong_to_cl = np.where(np.equal(self.cluster_assignment, cl))[0]
			# 		selected_indices = np.random.choice(a=belong_to_cl, size=ntodraw, replace=False)
			# 		perturbed_indices.extend(selected_indices.tolist())
			# 		# a single random shift
			# 		random_shift = random_direction_noise(npoints=1, ndim=n_features, magnitude=perturb_radius)
			# 		X_p[selected_indices, :] = np.vstack([xx + random_shift for xx in X_p[selected_indices, :]])
			# 		if len(perturbed_indices) >= npoints:
			# 			break

		if draw and (n_features in (1, 2)):
			plot_perturbation(X, X_p, perturbed_indices=perturbed_indices)

		return X_p


def IRPR_euc_distance(corpus1: Corpus, corpus2: Corpus, model: TextEmbedder = STTokenizerEmbedder()):
	embeddings1, embeddings2 = utils.get_corpora_embeddings(corpus1, corpus2, model)

	cosine = np.clip(cosine_similarity(embeddings1, embeddings2), -1, 1)
	# this is because sometimes cosine_similarity return values larger than 1
	table = np.arccos(cosine) / np.pi
	precision = np.nansum(np.nanmin(table, axis=1)) / table.shape[1]
	recall = np.nansum(np.nanmin(table, axis=0)) / table.shape[0]
	return 2 * (precision * recall) / (precision + recall)


def random_direction_noise(npoints=1, ndim=3, magnitude=1.0):
	# Shperical sampling
	# https://stackoverflow.com/questions/33976911/generate-a-random-sample-of-points-distributed-on-the-surface-of-a-unit-sphere
	vec = np.random.randn(ndim, npoints)
	vec /= np.linalg.norm(vec, axis=0)
	return magnitude * np.transpose(vec)


def gen_data(n_samples=100, n_features=2, cluster_std=2.0, jitter_std=0.1, pair_proportion=1.0, draw=True):
	# generate two samples where each element of X1 is close to its pair in X0 with proportion pair_proportion, but jittered by jitter_std
	# jitter_std should be small relative to cluster_std
	# centers=None means it will generate centers according to n_features

	X0 = BlobSample(n_samples=n_samples, n_features=n_features, cluster_std=cluster_std)
	X1, paired_indices = X0.gen_other(pair_proportion=pair_proportion, jitter_std=jitter_std)

	if draw and (n_features in (1, 2)):
		sns.set_theme(style="whitegrid")
		if n_features == 2:
			dx = X1[:, 0] - X0.X[:, 0]
			dy = X1[:, 1] - X0.X[:, 1]
			plt.scatter(X0.X[:, 0], X0.X[:, 1], c='blue', label='f_0', alpha=0.5)
			plt.scatter(X1[:, 0], X1[:, 1], edgecolors='orange', facecolors='none', label='f_1')
			# set aspect as equal
			plt.gca().set_aspect(1.0)
			if len(paired_indices):
				for xx, yy, dxx, dyy in zip(X0.X[paired_indices, 0], X0.X[paired_indices, 1], dx[paired_indices], dy[paired_indices]):
					plt.arrow(x=xx, y=yy, dx=dxx, dy=dyy, color='grey', linestyle="dotted")
				# label the last one
				plt.arrow(x=xx, y=yy, dx=dxx, dy=dyy, color='grey', linestyle="dotted", label="pairings")

		else:
			# use only first dimension so it uses the color attribute
			sns.kdeplot(X0.X[:,0], color='blue', label='f_0')
			sns.kdeplot(X1[:,0], color='orange', label='f_1')

		plt.legend()
		plt.title('Two distributions')
		plt.show()

	return X0, X1



# def gen_data(centers=None, n_samples=100, n_features=2, cluster_std=2.0, jitter_std=0.1, paired=True, draw=True):
# 	# generate two samples where each element of X1 is close to its pair in X0, but jittered by jitter_std
# 	# jitter_std should be small relative to cluster_std
# 	# centers=None means it will generate centers according to n_features
#
# 	X0, ass0, c0 = make_blobs(centers=centers, n_samples=n_samples, n_features=n_features, cluster_std=cluster_std, return_centers=True)
# 	n_features = X0.shape[1]
#
# 	if paired:
# 		# paired with X0
# 		cmat = np.diag([jitter_std] * n_features)
# 		X1 = np.vstack([np.random.multivariate_normal(mean=x, cov=cmat) for x in X0])
# 	else:
# 		# same iid as X0, using the same centers so that it is the same distribution
# 		clust_cmat = np.diag([cluster_std] * n_features)
# 		X1 = np.vstack([np.random.multivariate_normal(mean=x, cov=clust_cmat) for x in c0[ass0, :]])
#
# 	if draw and (n_features in (1, 2)):
# 		sns.set_theme(style="whitegrid")
# 		if n_features == 2:
# 			dx = X1[:, 0] - X0[:, 0]
# 			dy = X1[:, 1] - X0[:, 1]
# 			plt.scatter(X0[:, 0], X0[:, 1], c='blue', label='f_0', alpha=0.5)
# 			plt.scatter(X1[:, 0], X1[:, 1], c='orange', label='f_1', alpha=0.5)
# 			# set aspect as equal
# 			plt.gca().set_aspect(1.0)
# 			if paired:
# 				for xx, yy, dxx, dyy in zip(X0[:, 0], X0[:, 1], dx, dy):
# 					plt.arrow(x=xx, y=yy, dx=dxx, dy=dyy, color='grey', linestyle="dotted")
# 				plt.arrow(x=xx, y=yy, dx=dxx, dy=dyy, color='grey', linestyle="dotted", label='pairings')
#
# 		else:
# 			# use only first dimension so it uses the color attribute
# 			sns.kdeplot(X0[:,0], color='blue', label='f_0', alpha=0.5)
# 			sns.kdeplot(X1[:,0], color='orange', label='f_1', alpha=0.5)
#
# 		plt.legend()
# 		plt.title('Paired distribution' if paired else 'Two distributions')
# 		plt.show()
#
# 	return X0, X1



def plot_perturbation(X, X_p, perturbed_indices=None):
	# X is original matrix
	# X_p is the perturbed
	# perturbed_indices is the indices that were changed between X and X_p

	assert X.shape == X_p.shape
	n_features = X.shape[1]
	assert n_features in (1,2)

	sns.set_theme(style="whitegrid")
	if perturbed_indices is None:
		perturbed_indices = []

	if len(perturbed_indices) < len(X):
		not_selected = np.array([ii for ii in range(len(X)) if ii not in perturbed_indices])
		plt.scatter(X[not_selected, 0], X[not_selected, 1], c='orange', label='untouched f_1', alpha=0.5)

	xo = X[perturbed_indices, 0]
	xn = X_p[perturbed_indices, 0]
	dx = xn - xo

	if n_features == 2:
		yo = X[perturbed_indices, 1]
		yn = X_p[perturbed_indices, 1]
		dy = yn - yo

		plt.scatter(xn, yn, s=80, facecolors='none', edgecolors='green', label='perturbed f_1', alpha=0.5)
		plt.scatter(xo, yo, s=80, c='green', label='original f_1', alpha=0.5)
		# set aspect as equal
		plt.gca().set_aspect(1.0)

		for xx, yy, dxx, dyy in zip(xo, yo, dx, dy):
			plt.arrow(x=xx, y=yy, dx=dxx, dy=dyy, color='green')

	else:
		sns.kdeplot(X[:, 0], color='orange', label='original f_1')
		sns.kdeplot(X_p[:, 0], color='green', label='perturbed f_1')

	plt.title('Perturbed distribution')

	plt.legend()
	plt.show()


def perturb_points(X, ppoints=0.1, perturb_std=3.0, perturb_radius=0, draw=False):
	npoints = ppoints_to_n(p=ppoints, X=X)
	n_features = X.shape[1]

	# assert not((perturb_std > 0) and (perturb_radius > 0)), 'Only one of perturb_std and perturb_radius can be nonzero'

	# perturb these by a random draw around their value
	X_p = deepcopy(X)

	if npoints == 0:
		sel_pts = np.array([])
	else:
		sel_pts = np.random.choice(a=X.shape[0], replace=False, size=npoints)
		if perturb_std > 0:
			# white-noise multivariate normal
			noise = np.random.multivariate_normal(mean=[0] * n_features, cov=np.diag([perturb_std] * n_features), size=npoints)
		else:
			noise = random_direction_noise(npoints=npoints, ndim=n_features, magnitude=perturb_radius)
		X_p[sel_pts, :] = X_p[sel_pts, :] + noise

	if draw and (n_features in (1, 2)):
		plot_perturbation(X, X_p, perturbed_indices=sel_pts)

	return X_p


def run_distributional_comparison_group_shift(metrics, metric_names, X0, X1, perturb_params={"npoints": [1,2,3], "perturb_radius": 0.1}, force_away=True, repetitions=5):
	distance_results = []

	assert isinstance(X0, BlobSample) # not ndarray since need it for the
	assert not np.any([vv in [zipf_distance, chi_square_distance] for vv in metrics]), 'metrics cannot be one of zipf_distance or chi_square_distance'
		# this runs on
	assert isinstance(perturb_params, dict)
	param_names = ['ppoints', 'perturb_radius']
	assert all([vv in perturb_params for vv in param_names])
	is_varying = {kk: len(vv) > 1 if isinstance(vv, list) else False for kk, vv in perturb_params.items()}
	assert sum([vv for vv in is_varying.values()]) == 1, 'exactly one entry in perturb_params must be a list of length > 1 '
	assert len(metrics) == len(metric_names), 'The number of metrics and metric names must be equal'

	nfeatures = X1.shape[1]
	pp = 0 # doesn't matter since not varying
	psd = 0

	# use the same draws
	if is_varying["ppoints"]:
		for step, ppts in enumerate(perturb_params["ppoints"]):
			# generate multiple repetitions of a random perturbation of the same degree
			X1_p = [X0.perturb_grouped_points(X1=X1, ppoints=ppts, perturb_radius=perturb_params["perturb_radius"], force_away=force_away, draw=False)
					for rr in range(repetitions)]

			for metfunc, mn in zip(metrics, metric_names):
				distance_results.append(np.vstack([[step, mn, rr, nfeatures, ppts, psd, perturb_params["perturb_radius"], pp,
													metfunc(corpus1=X0.X, corpus2=X1rep, model=None)]
						  for rr, X1rep in enumerate(X1_p)]))
	else:
		# radius
		for step, radius in enumerate(perturb_params["perturb_radius"]):
			# generate multiple repetitions of a random perturbation of the same degree
			X1_p = [X0.perturb_grouped_points(X1=X1, ppoints=perturb_params["ppoints"], perturb_radius=radius, force_away=force_away, draw=False)
					for rr in range(repetitions)]

			for metfunc, mn in zip(metrics, metric_names):
				distance_results.append(np.vstack([[step, mn, rr, nfeatures, perturb_params["ppoints"], psd, radius, pp,
													metfunc(corpus1=X0.X, corpus2=X1rep, model=None)]
												   for rr, X1rep in enumerate(X1_p)]))

	distance_results = pd.DataFrame(np.vstack(distance_results), columns=['step', 'metric', 'rep', 'nfeatures', 'ppoints', 'perturb_std', 'perturb_radius', 'paired', 'value'])

	return distance_results.astype({'step': int, 'value': float, 'nfeatures': int, 'ppoints': float, 'metric': 'category',
									'perturb_std': float, 'perturb_radius': float, 'paired': float, 'rep': int})



# similar to runKSC
def run_distributional_comparison_n_or_sd(metrics, metric_names, X0, X1, perturb_params={"ppoints": [0.1, 0.2, 0.3], "perturb_std": 3.0}, repetitions=5):
	distance_results = []

	assert all([isinstance(xx, np.ndarray) for xx in [X0, X1]])
	assert not np.any([vv in [zipf_distance, chi_square_distance] for vv in metrics]), 'metrics cannot be one of zipf_distance or chi_square_distance'
		# this runs on
	assert isinstance(perturb_params, dict)
	assert 'ppoints' in perturb_params
	assert sum([vv in perturb_params for vv in ['perturb_std', 'perturb_radius']]), 'Exactly one of perturb_std, perturb_radius must be in perturb_params'
	is_varying = {kk: len(vv) > 1 if isinstance(vv, list) else False for kk, vv in perturb_params.items()}
	assert sum([vv for vv in is_varying.values()]) == 1, 'exactly one entry in perturb_params must be a list of length > 1 '
	assert len(metrics) == len(metric_names), 'The number of metrics and metric names must be equal'

	perturb_type = 'std' if 'perturb_std' in perturb_params else 'radius'
	pp = 1.0 # paired proportions (full by default)
	nfeatures = X0.shape[0]

	# use the same draws
	if is_varying["ppoints"]:
		psd = perturb_params['perturb_std'] if perturb_type == 'std' else 0
		prad = perturb_params['perturb_radius'] if perturb_type == 'radius' else 0

		for step, ppts in enumerate(perturb_params["ppoints"]):
			# generate multiple repetitions of a random perturbation of the same degree
			if perturb_type == 'std':
				X1_p = [perturb_points(X=X1, ppoints=ppts, perturb_std=perturb_params["perturb_std"], draw=False)
						for rr in range(repetitions)]
			else:
				X1_p = [perturb_points(X=X1, ppoints=ppts, perturb_radius=perturb_params["perturb_radius"], draw=False)
						for rr in range(repetitions)]

			for metfunc, mn in zip(metrics, metric_names):
				distance_results.append(np.vstack([[step, mn, rr, nfeatures, ppts, psd, prad, pp, metfunc(corpus1=X0, corpus2=X1rep, model=None)]
						  for rr, X1rep in enumerate(X1_p)]))
	else:
		if 'perturb_std' in perturb_params:
			for step, sd in enumerate(perturb_params["perturb_std"]):
				# generate multiple repetitions of a random perturbation of the same degree
				X1_p = [perturb_points(X=X1, ppoints=perturb_params["ppoints"], perturb_std=sd, draw=False)
						for rr in range(repetitions)]

				for metfunc, mn in zip(metrics, metric_names):
					distance_results.append(np.vstack([[step, mn, rr, nfeatures, perturb_params["ppoints"], sd, 0, pp, metfunc(corpus1=X0, corpus2=X1rep, model=None)]
													   for rr, X1rep in enumerate(X1_p)]))
		else:
			# radius
			for step, radius in enumerate(perturb_params["perturb_radius"]):
				# generate multiple repetitions of a random perturbation of the same degree
				X1_p = [perturb_points(X=X1, ppoints=perturb_params["ppoints"], perturb_radius=radius, perturb_std=0, draw=False)
						for rr in range(repetitions)]

				for metfunc, mn in zip(metrics, metric_names):
					distance_results.append(np.vstack([[step, mn, rr, nfeatures, perturb_params["ppoints"], 0, radius, pp,
														metfunc(corpus1=X0, corpus2=X1rep, model=None)]
													   for rr, X1rep in enumerate(X1_p)]))

	distance_results = pd.DataFrame(np.vstack(distance_results), columns=['step', 'metric', 'rep', 'nfeatures', 'ppoints', 'perturb_std', 'perturb_radius', 'paired', 'value'])

	return distance_results.astype({'step': int, 'value': float, 'nfeatures': int, 'ppoints': float, 'metric': 'category',
									'perturb_std': float, 'perturb_radius': float, 'paired': float, 'rep': int})



def run_distributional_comparison_nfeatures(metrics, metric_names, gen_func=None, perturb_func=None, repetitions=5,
											nfeatures_vec=[2,3,4]):

	if gen_func is None:
		# make n_features variable
		gen_func = lambda nf: gen_data(n_features=nf)
		sig = signature(gen_data)
	else:
		sig = signature(gen_func)

	pp = sig.parameters['pair_proportion'].default if 'pair_proportion' in sig.parameters else 1.0

	if perturb_func is None:
		perturb_func = lambda X: perturb_points(X=X, draw=False)
		# get the defaults from perturb_points
		sig = signature(perturb_points)
	else:
		# the function must specify ppoints and perturb_std
		sig = signature(perturb_func)

	perturb_std = sig.parameters['perturb_std'].default if 'perturb_std' in sig.parameters else 0.0
	perturb_radius = sig.parameters['perturb_radius'].default if 'perturb_radius' in sig.parameters else 0.0
	ppoints = sig.parameters['ppoints'].default
	# determine if needs the source X0 or just target X
	needs_orig_X = all([kk in sig.parameters for kk in ['X0', 'X1']])

	distance_results = []

	assert not np.any([vv in [zipf_distance, chi_square_distance] for vv in
					   metrics]), 'metrics cannot be one of zipf_distance or chi_square_distance'
	# this runs on
	assert len(metrics) == len(metric_names), 'The number of metrics and metric names must be equal'

	for step, nf in enumerate(nfeatures_vec):
		# generate multiple repetitions of a random perturbation of the same degree
		# Xs = [gen_func(nf) for rr in range(repetitions)]
		Xs = gen_func(nf)
		if needs_orig_X:
			Xs = [(Xs[0].X, perturb_func(X0=Xs[0], X1=Xs[1])) for rr in range(repetitions)]
		else:
			Xs = [(Xs[0].X, perturb_func(Xs[1])) for rr in range(repetitions)]

		for metfunc, mn in zip(metrics, metric_names):
			distance_results.append(np.vstack([[step, mn, rr, nf, ppoints, perturb_std, perturb_radius, pp,
												metfunc(corpus1=xx[0], corpus2=xx[1], model=None)]
												   for rr, xx in enumerate(Xs)]))

	distance_results = pd.DataFrame(np.vstack(distance_results),
									columns=['step', 'metric', 'rep', 'nfeatures', 'ppoints', 'perturb_std', 'perturb_radius', 'paired', 'value'])

	return distance_results.astype({'step': int, 'value': float, 'nfeatures': int, 'ppoints': float, 'metric': 'category',
									'perturb_std': float, 'perturb_radius': float, 'paired': float, 'rep': int})


def run_distributional_comparison_pairing(metrics, metric_names, gen_func=None,  repetitions=5, pair_proportions=[0, 1/3, 2/3, 1]):
	# change the pairing_proportion parameter

	if gen_func is None:
		# make pair_proportion variable
		gen_func = lambda pp: gen_data(pair_proportion=pp)

	distance_results = []

	assert not np.any([vv in [zipf_distance, chi_square_distance] for vv in
					   metrics]), 'metrics cannot be one of zipf_distance or chi_square_distance'
	# this runs on
	assert len(metrics) == len(metric_names), 'The number of metrics and metric names must be equal'

	# irrelevant parameters
	perturb_std = 0.0
	ppoints = 0.0
	perturb_radius = 0.0
	nfeatures = 2

	for step, pp in enumerate(pair_proportions):
		# generate multiple repetitions of a random perturbation of the same degree
		Xs = [gen_func(pp) for rr in range(repetitions)]

		for metfunc, mn in zip(metrics, metric_names):
			distance_results.append(np.vstack([[step, mn, rr, nfeatures, ppoints, perturb_std, perturb_radius, pp,
												metfunc(corpus1=xx[0].X, corpus2=xx[1], model=None)]
												   for rr, xx in enumerate(Xs)]))

	distance_results = pd.DataFrame(np.vstack(distance_results),
									columns=['step', 'metric', 'rep', 'nfeatures', 'ppoints', 'perturb_std', 'perturb_radius', 'paired', 'value'])

	return distance_results.astype({'step': int, 'value': float, 'nfeatures': int, 'ppoints': float, 'metric': 'category',
									'perturb_std': float, 'perturb_radius': float, 'paired': float, 'rep': int})


def plot_distributional_comparison(distance_results_df, boxplot=True, decimals=2,
								   add_line=True, showfliers=True, ncolumns=6):
	# no normalization here
	uv = {kk: vv.nunique() for kk, vv in distance_results_df[['nfeatures', 'ppoints', 'perturb_std', 'perturb_radius', 'paired']].items()}
	uv = {kk: vv for kk, vv in uv.items() if vv > 1}
	# which parameter is changing
	varying_param = list(uv.keys())[0]

	var_axis = distance_results_df[['step', varying_param]].drop_duplicates().sort_values(by='step')
	if pd.api.types.is_float_dtype(var_axis.dtypes.loc[varying_param]):
		# round
		var_axis[varying_param] = var_axis[varying_param].round(decimals=decimals)

	sns.set_theme(style="whitegrid")
	sns.set(font_scale=1.3)
	nmetrics = distance_results_df['metric'].nunique()
	ncolumns = min(ncolumns, nmetrics)
	nrows = int(np.ceil(nmetrics/ncolumns))
	# number of empty cells
	nempty = (nrows * ncolumns) - nmetrics

	fig, axs = plt.subplots(nrows, ncolumns, figsize=(6*ncolumns, 5*nrows))
	# sort results by metric name
	metric_order = sorted(distance_results_df['metric'].unique().tolist(), key=sort_numsuffix)
	distance_results_df['metric'] = distance_results_df['metric'].cat.set_categories(metric_order, ordered=True)

	# distance_results_df.sort_values(by=['metric'], inplace=True, key=lambda col: col.map(sort_numsuffix))

	for ax, res in zip(fig.axes, distance_results_df.groupby('metric')):
		if boxplot:
			sns.boxplot(x='step', y='value', data=res[1], ax=ax, color='orange', showfliers=showfliers)
			if add_line:
				sns.regplot(x='step', y='value', data=res[1], ax=ax, scatter=False)
			# grp_df = res[1].groupby(varying_param)
			# uvals = sorted(list(grp_df.keys()))
			# for val, df_byval in res[1].groupby(varying_param):
			# 	bp = plt.boxplot(df_byval['value'], positions=[val])
			# plt.xticks(uvals, uvals)
			# plt.autoscale()
		else:
			sns.regplot(x='step', y='value', data=res[1], ax=ax,
						scatter_kws={'color': 'orange'})
		ax.set_title('{}'.format(res[0]))
		ax.set_xlabel(varying_param)

		ax.set_xticks(ticks=list(range(len(var_axis))), labels=var_axis[varying_param])

	for idx in range(1, nempty+1):
		# get rid of empty frames at the end
		axs.flat[-idx].set_visible(False)

	plt.subplots_adjust(left=0.05, bottom=0.1, right=0.99, top=0.9,
						wspace=0.3,	hspace=0.4)

		# fname = 'ksc_results/images/{}_{}{}.png'.format('boxplot' if boxplot else 'scatter',
		# 												'standard' if standardized else 'raw', suffix)
		# plt.savefig(fname)

	plt.show()

# if __name__ == '__main__':
# 	X0, X1 = gen_data()
# 	X0.perturb_grouped_points(X1=X1, npoints=50, perturb_radius=1)