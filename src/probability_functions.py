from scipy.stats import hypergeom
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from scipy.special import comb
from collections import defaultdict

SEMANTIC_DIST = lambda a, b: 0.0 if a == b else (0.5 if a + b == 0 else 1.0)
NONSEMANTIC_DIST = lambda a, b: 0.0 if a == b else 1.0
from sklearn.metrics import pairwise_distances

def hgprobs(sz, n, nsame=None, nunq=True):
	# The hypergeometric distribution models drawing objects from a bin. M is the total number of objects, n is total number of Type I objects. The random variate represents the number of Type I objects in N drawn without replacement from the total population.
	# hypergeom(M, n, N)
	x = np.array(list(range(0, sz + 1))).astype(int)

	if nsame is None:
		# the other nsegments - sz are potentially the same type, want to test how many duplicates
		nsame = n - sz
	p = np.array([hypergeom.pmf(xx, n, nsame, sz) for xx in x])

	# number of unique items in double lottery where S1 is of size sz, or number of items in S1 that are in S2
	return n - x if nunq else x, p


# if distance is distributional and semantic
def duplicate_distance(i, j, n=10):
	i = np.clip(a=int(i), a_min=0, a_max=n)
	j = np.clip(a=int(j), a_min=0, a_max=n)

	# number of unique items in each of S^i and S^j, with respective probabilities
	nunq_j, pj = hgprobs(sz=j, n=n)
	nunq_i, pi = hgprobs(sz=i, n=n)

	# number of items in intersection, not number of unique items in the total sample
	tmp = [[hgprobs(sz=ii, nsame=jj, n=n, nunq=False) for ii in nunq_i]
		   for jj in nunq_j]
	nunq_ij = [[vvv[0] for vvv in vv] for vv in tmp]  # =z_ij
	p_ij = [[vvv[1] for vvv in vv] for vv in tmp]
	# now multiply conditionals to get the full unconditional probability
	for jj, ppj in enumerate(pj):
		for ii, ppi in enumerate(pi):
			p_ij[jj][ii] = p_ij[jj][ii] * ppj * ppi

	pu = defaultdict(list)
	for nn, pp in zip(nunq_ij, p_ij):
		for nnn, ppp in zip(nn, pp):
			for nnnn, pppp in zip(nnn, ppp):
				pu[nnnn].append(pppp)
	pu = {kk: sum(vv) for kk, vv in pu.items()}

	return (nunq_i, pi), (nunq_j, pj), pu


def double_lottery(i, n, C0, C1):
	i = np.clip(a=int(i), a_min=0, a_max=n)
	return np.append(arr=np.random.choice(a=C0, size=i, replace=False),
					 values=np.random.choice(a=C1, size=n - i, replace=False))


def simulate_double_lottery_intersection(i, j, n, C0, C1, nsamp=1000, semantic=True):

	Sj = np.array([double_lottery(i=j, n=n, C0=C0, C1=C1) for rr in range(nsamp)])
	Si = np.array([double_lottery(i=i, n=n, C0=C0, C1=C1) for rr in range(nsamp)])
	# isec = np.array([len(np.intersect1d(ar1=aa, ar2=bb)) for aa, bb in zip(Si, Sj)])
	isec = np.array([intersection_distance(x=aa, y=bb, semantic=semantic, as_distance=False) for aa, bb in zip(Si, Sj)])
	c = Counter(isec)
	# probabilities of various intersection sizes
	c = {kk: vv / nsamp for kk, vv in c.items()}
	c.update({kk: 0.0 for kk in range(n + 1) if kk not in c})
	return c


# def euclidean(x, y, nsegments=10):
#     # x is an input point in one sample
#     # y is a sample
#     d = np.abs(x - y)
#     # maximum distance wraps
#     # d[ d == (nsegments-1)] = 0
#     return d

# def circular_euclidean(x, y, nsegments=10):
#     a = euclidean(x, y, nsegments)
#     # wrap the distance (e.g., 1 to 8 when nsegments=10, 8 can be like -1 =(8 - (10-1)), so distance =2)
#     b = euclidean(x, y - nsegments, nsegments)
#     return np.minimum(a, b)

# def hausdorff(x, y, nsegments=10, as_max=True):
#     # min--distance from each item to its closest neighbor
#     d = np.array([circular_euclidean(xx, y, nsegments).min() for xx in x])
#     return d.max() if as_max else d.mean()

# def hausdorff_metric(x, y, nsegments=10, as_max=True):
#     # make symmetric
#     d1 = hausdorff(x, y, nsegments, as_max)
#     d2 = hausdorff(y, x, nsegments, as_max)
#     return max(d1, d2)

# def hausdorff_metric(x, y, n=10, as_max=True):
#     # an item has 0 nin distance to another sample if it exists in the sample (distance 0), otherwise 1
#     # thus the hausdorff metric (the max of the minimum neighbor distances) is 0 (if all neighborhood distances are 0, hence the sets equal each other)
#     # otherwise 1 if at least one item doesn't appear
#     d1 = 1 - np.isin(x, y).astype(int)
#     d2 = 1 - np.isin(y, x).astype(int)
#     d1 = d1.max() if as_max else d1.mean()
#     d2 = d2.max() if as_max else d2.mean()

#     return max(d1, d2) #0 if (set(x) == set(y)) else 1

def intersection_distance(x, y, semantic=True, as_distance=False):
	# if not as_distance, return just intersection size
	if semantic:
		x = np.abs(x)
		y = np.abs(y)
	xu = np.unique(x)
	yu = np.unique(y)
	lx = len(xu)
	ly = len(yu)
	uxy = np.union1d(xu, yu)
	isize = len(np.intersect1d(ar1=xu, ar2=yu))
	return 1 - (isize/max([len(x), len(y)])) if as_distance else isize
	# return 1 - (isize/len(uxy)) if as_distance else isize

def energy_distance(x, y, semantic=True, normalize=False):
	dxy = SEMANTIC_DIST if semantic else NONSEMANTIC_DIST
	x = x.reshape(-1,1)
	y = y.reshape(-1,1)
	between = pairwise_distances(X=x, Y=y, metric=dxy)
	within1 = pairwise_distances(X=x, metric=dxy)
	within2 = pairwise_distances(X=y, metric=dxy)

	# if semantic:
	# 	x = np.abs(x)
	# 	y = np.abs(y)
	#
	# def internal_binary(a, b):
	# 	d = np.subtract.outer(a, b) # pairwise differences
	# 	d[ d != 0 ] = 1.0 # convert 0s to 0 and others to 1
	# 	return d
	#
	# between = internal_binary(x, y)
	# within1 = internal_binary(x, x)
	# within2 = internal_binary(y, y)
	A2 = 2 * between.mean()
	B = within1.mean()
	C = within2.mean()

	edist = A2 - B - C
	return edist/A2 if normalize else np.sqrt(edist)

def average_hausdorff_distance(x, y, semantic=True):
	# an item has 0 min distance to another sample if it exists in the sample (distance 0), otherwise 1
	# the sum of these is equal to the intersection length
	# if semantic:
	# 	x = np.abs(x)
	# 	y = np.abs(y)
	# x_in_y = np.isin(x, y).mean()
	# y_in_x = np.isin(y, x).mean()
	# 
	# return 0.5 * ((1 - x_in_y) + (1 - y_in_x))
	dxy = SEMANTIC_DIST if semantic else NONSEMANTIC_DIST
	x = x.reshape(-1,1)
	y = y.reshape(-1,1)
	between = pairwise_distances(X=x, Y=y, metric=dxy)

	d_from_y = np.nanmin(between, axis=0).mean()
	d_from_x = np.nanmin(between, axis=1).mean()
	return np.mean([d_from_y, d_from_x])

def simulate_double_lottery_hausdorff(i, j, n, C0, C1, as_max=True, nsamp=1000, semantic=True):
	# semantic will have C0=C1
	# nonsemantic will have non-overlapping C0, C1

	Sj = np.array([double_lottery(i=j, n=n, C0=C0, C1=C1) for rr in range(nsamp)])
	Si = np.array([double_lottery(i=i, n=n, C0=C0, C1=C1) for rr in range(nsamp)])

	# multiply by n so is integer
	h = np.array([1 - average_hausdorff_distance(xx, yy, semantic=semantic) for xx, yy in zip(Si, Sj)]) * n
	h = np.round(h, 1)
	c = Counter(h)
	# probabilities of various intersection sizes
	c = {kk: vv / nsamp for kk, vv in c.items()}
	half_steps = np.round(np.linspace(start=0, stop=n, num=2 * n + 1, dtype=float), 1)

	c.update({kk: 0.0 for kk in half_steps if kk not in c})
	return c


# def palindrome_index(nsegments):
#     nsegments = max(3, int(nsegments))
#     # i = np.clip(a=np.array(i).astype(int), a_min=0, a_max=nsegments)
#     m = int(np.floor(nsegments/2))
#     half = np.linspace(start=0, stop=m, num=m+1)
#     # event so end of half is repeated, and reverse, but drop last element
#     return np.append(half, half[-(2 if (nsegments % 2) == 0 else 1)::-1])


def E_intersection_semantic_dist(i, j, n):
	ni = n - i
	nj = n - j
	# return #nsegments - (1/nsegments) * (i*ni + j*nj - (i*j)*ni*nj/(nsegments**2) )
	return (1 / n) * (n - i * ni / n) * (n - j * nj / n)


def E_intersection_nonsemantic_dist(i, j, n):
	ni = n - i
	nj = n - j
	# return #nsegments - (1/nsegments) * (i*ni + j*nj - (i*j)*ni*nj/(nsegments**2) )
	return (i * j + (n - j) * (n - i)) / n


def E_intersection_nonsemantic_nondist(i, j, n):
	ni = n - i
	nj = n - j
	# return #nsegments - (1/nsegments) * (i*ni + j*nj - (i*j)*ni*nj/(nsegments**2) )
	return n - (i * j + (n - j) * (n - i)) / n


def distr_of_indep_xsum(x1, x2, p1, p2, trim=False):
	# calculate probablities of sums of x1, x2 happening independently
	mx = x1.max() + x2.max()
	# consider values from 0 to mx so can just use index
	xvals = np.arange(mx + 1)
	pvals = np.zeros(mx + 1)
	for xx1, pp1 in zip(x1, p1):
		for xx2, pp2, in zip(x2, p2):
			xsum = xx1 + xx2
			# pvals[xsum] += (comb(n, i)*pp1 + comb(n, j)*pp2)
			# independent probabilities are multiplied
			pvals[xsum] += (pp1 * pp2)
	if trim:
		max_nonzero = np.where(pvals > 0)[0].max()
		xvals = xvals[:(max_nonzero + 1)]
		pvals = pvals[:(max_nonzero + 1)]

	return xvals, pvals / pvals.sum()


def distr_of_indep_xsum_frac(x1, x2, p1, p2, n1, n2):
	sel1 = x1 <= n1
	sel2 = x2 <= n2

	x1 = x1[sel1] / n1
	p1 = p1[sel1]
	x2 = x2[sel2] / n2
	p2 = p2[sel2]

	if len(x1) == 0:
		pvals = dict(zip(x1, p1))
	elif len(x2) == 0:
		pvals = dict(zip(x2, p2))
	else:
		pvals = defaultdict(list)
		for xx1, pp1 in zip(x1, p1):
			for xx2, pp2, in zip(x2, p2):
				xsum = (xx1 + xx2) / 2
				pvals[xsum].append(pp1 * pp2)
		pvals = {kk: sum(vv) for kk, vv in pvals.items()}

	return pvals


def probs_intersection_nonsemantic_dist(i, j, n):
	Z_1ij, p1 = hgprobs(sz=j, n=n, nsame=i, nunq=False)
	Z_2ij, p2 = hgprobs(sz=n - j, n=n, nsame=n - i, nunq=False)

	return distr_of_indep_xsum(x1=Z_1ij, x2=Z_2ij, p1=p1, p2=p2)


def probs_intersection_nonsemantic_nondist(i, j, n):
	Z_1ij, p1 = hgprobs(sz=j, n=n, nsame=i, nunq=False)
	Z_2ij, p2 = hgprobs(sz=n - j, n=n, nsame=n - i, nunq=False)

	xvals, pvals = distr_of_indep_xsum(x1=Z_1ij, x2=Z_2ij, p1=p1, p2=p2)
	# nondist is n - sum, so reverse the order of x
	return xvals[::-1], pvals


def probs_intersection_semantic_nondist(i, j, n):
	# number of unique items in each of S^i and S^j, with respective probabilities
	nunq_j, punq_j = hgprobs(sz=j, n=n)
	nunq_i, punq_i = hgprobs(sz=i, n=n)

	# conditional on S^j having nuj unique items, what's the probability of, out of ii or n-ii unique items, that x of the ii or n-ii are
	# in the nuj unique items
	# consider from each direction
	Z_i2j_1 = [hgprobs(sz=i, nsame=nuj, n=n, nunq=False) for nuj in nunq_j]
	# print(Z_i2j_1)
	Z_i2j_2 = [hgprobs(sz=n - i, nsame=nuj, n=n, nunq=False) for nuj in nunq_j]
	# print(Z_i2j_1)

	Z_j2i_1 = [hgprobs(sz=j, nsame=nui, n=n, nunq=False) for nui in nunq_i]
	# print(Z_j2i_1)
	Z_j2i_2 = [hgprobs(sz=n - j, nsame=nui, n=n, nunq=False) for nui in nunq_i]
	# print(Z_j2i_2)

	# key is number of items, value is list of probabilities of how it can happen (which are then summed)
	Z_i2j_probs = defaultdict(list)
	Z_j2i_probs = defaultdict(list)

	for pj, zi1, zi2 in zip(punq_j, Z_i2j_1, Z_i2j_2):
		# the total number of common items (including duplicates) is Z^i_1 + Z^i_2
		Zi, pzi = distr_of_indep_xsum(x1=zi1[0], x2=zi2[0], p1=zi1[1], p2=zi2[1])

		for z, p in zip(Zi, pzi):
			# multiply conditional by initial probability of having that many unique values in j
			Z_i2j_probs[z].append(p * pj)

	# now repeat in the reverse order
	for pi, zj1, zj2 in zip(punq_i, Z_j2i_1, Z_j2i_2):
		Zj, pzj = distr_of_indep_xsum(x1=zj1[0], x2=zj2[0], p1=zj1[1], p2=zj2[1])

		for z, p in zip(Zj, pzj):
			# multiply conditional by initial probability of having that many unique values in i
			Z_j2i_probs[z].append(p * pi)

		# now sum
	xvals = np.arange(n + 1)
	pZi = np.array([sum(Z_i2j_probs[xx]) for xx in xvals])
	pZj = np.array([sum(Z_j2i_probs[xx]) for xx in xvals])

	# now get distribution of the sum, and then divide xvals by 2 to get it in the range 0--n, (then by n to get a fraction)
	Zintersection, pintersection = distr_of_indep_xsum(x1=xvals, x2=xvals, p1=pZi, p2=pZj)
	# print(pintersection.sum())
	# return distr_of_indep_xsum_frac(x1=xvals, x2=xvals, p1=pZi, p2=pZj, n1=, n2)
	return Zintersection / (n * 2), pintersection / pintersection.sum()